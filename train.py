import os
import sys
import json
import time
import logging
import gc
import struct
import argparse
from pathlib import Path

import cv2
import numpy as np
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoImageProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer
from nitrogen.shared import BUTTON_ACTION_TOKENS

from torchvision import transforms
from PIL import Image

try:
    from turbojpeg import TurboJPEG
    JPEG_DECODER = TurboJPEG()
    HAS_TURBO = True
except ImportError:
    HAS_TURBO = False
    print("⚠️ TurboJPEG not found, using cv2 (slower)")

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = os.path.expanduser("~/my_dataset")
CHECKPOINT_PATH = os.path.expanduser("ng.pt")

BATCH_SIZE = 96
EPOCHS = 10
VAL_INTERVAL = 2

USE_ONECYCLE = True

ONECYCLE_MAX_LR = 1.2e-5
ONECYCLE_PCT_START = 0.15
ONECYCLE_DIV_FACTOR = 25
ONECYCLE_FINAL_DIV_FACTOR = 1000

COSINE_LR = 1e-5
COSINE_WARMUP_RATIO = 0.1

EMA_DECAY = 0.9999
GRAD_CLIP = 1.0
VAL_SPLIT = 0.1

USE_BFLOAT16 = True
USE_TORCH_COMPILE = True
COMPILE_MODE = "default"

NUM_WORKERS = 6
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True
PIN_MEMORY = True

MAX_CHECKPOINTS = 3
LOG_FILE = "training_wsl.log"

# ============================================
# INITIALIZATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ============================================
# ARGS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="NitroGen Training")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--resume-best', action='store_true', help='Resume from best_model.pt')
    parser.add_argument('--resume-latest', action='store_true', help='Resume from latest checkpoint')
    return parser.parse_args()


def find_latest_checkpoint():
    checkpoints = list(Path(".").glob("checkpoint_ep*.pt"))
    if not checkpoints:
        return None
    
    def get_epoch_num(path):
        try:
            s = path.stem.replace("checkpoint_ep", "")
            return int(s) if s.isdigit() else 0
        except ValueError:
            return 0
    
    return str(max(checkpoints, key=get_epoch_num))

# ============================================
# SCHEDULER STATE
# ============================================

def build_scheduler(optimizer, total_steps):
    if USE_ONECYCLE:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=ONECYCLE_MAX_LR,
            total_steps=total_steps,
            pct_start=ONECYCLE_PCT_START,
            div_factor=ONECYCLE_DIV_FACTOR,
            final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
            anneal_strategy='cos'
        )
        scheduler_meta = {
            "scheduler_type": "onecycle",
            "onecycle_max_lr": ONECYCLE_MAX_LR,
            "onecycle_pct_start": ONECYCLE_PCT_START,
            "onecycle_div_factor": ONECYCLE_DIV_FACTOR,
            "onecycle_final_div_factor": ONECYCLE_FINAL_DIV_FACTOR,
        }
    else:
        warmup_steps = int(total_steps * COSINE_WARMUP_RATIO)
        s1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])
        scheduler_meta = {
            "scheduler_type": "cosine_warmup",
            "cosine_lr": COSINE_LR,
            "cosine_warmup_ratio": COSINE_WARMUP_RATIO,
            "warmup_steps": warmup_steps,
        }

    return scheduler, scheduler_meta


def make_training_state(steps_per_epoch, total_steps, scheduler_meta):
    state = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "use_onecycle": USE_ONECYCLE,
    }
    state.update(scheduler_meta)
    return state


def validate_resume_compatibility(checkpoint, expected_training_state):
    missing = [k for k in ("optimizer", "scheduler") if k not in checkpoint]
    if missing:
        raise ValueError(
            "Checkpoint is missing resume state: "
            + ", ".join(missing)
            + ". Use a fresh base checkpoint or a newer training checkpoint."
        )

    saved_state = checkpoint.get("training_state")
    if saved_state is None:
        if USE_ONECYCLE:
            raise ValueError(
                "This checkpoint predates resumable OneCycle support and cannot be safely resumed."
            )
        return

    mismatches = []
    guarded_keys = [
        "scheduler_type",
        "epochs",
        "steps_per_epoch",
        "total_steps",
        "use_onecycle",
    ]

    if USE_ONECYCLE:
        guarded_keys.extend([
            "onecycle_max_lr",
            "onecycle_pct_start",
            "onecycle_div_factor",
            "onecycle_final_div_factor",
        ])
    else:
        guarded_keys.extend([
            "cosine_lr",
            "cosine_warmup_ratio",
        ])

    for key in guarded_keys:
        saved_value = saved_state.get(key)
        expected_value = expected_training_state.get(key)
        if saved_value is None:
            continue
        if saved_value != expected_value:
            mismatches.append(f"{key}: saved={saved_value}, current={expected_value}")

    if mismatches:
        raise ValueError(
            "Unsafe resume blocked because the training schedule changed:\n- "
            + "\n- ".join(mismatches)
        )

# ============================================
# DATASET HELPERS
# ============================================

def build_line_index(filepath):
    index_path = filepath + ".idx"
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            count = struct.unpack('I', f.read(4))[0]
            offsets = struct.unpack(f'{count}Q', f.read(8 * count))
        return list(offsets)
    
    offsets = []
    with open(filepath, 'rb') as f:
        while True:
            pos = f.tell()
            if not f.readline():
                break
            offsets.append(pos)
    
    with open(index_path, 'wb') as f:
        f.write(struct.pack('I', len(offsets)))
        f.write(struct.pack(f'{len(offsets)}Q', *offsets))
    
    return offsets


def read_json_line(filepath, offset):
    with open(filepath, 'rb') as f:
        f.seek(offset)
        return json.loads(f.readline())

# ============================================
# CHECKPOINTS HELPERS
# ============================================

def get_model_state_dict(model):
    """Extract state_dict from the model, including torch.compile wrappers."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def get_ema_state_dict(ema_model):
    """Extract state_dict from an AveragedModel wrapper."""
    if hasattr(ema_model, 'module'):
        inner = ema_model.module
        if hasattr(inner, '_orig_mod'):
            return inner._orig_mod.state_dict()
        return inner.state_dict()
    return ema_model.state_dict()


def get_model_attr(model, attr, default=None):
    """Safely read a model attribute, including torch.compile wrappers."""
    if hasattr(model, '_orig_mod'):
        return getattr(model._orig_mod, attr, default)
    return getattr(model, attr, default)

# ============================================
# VALIDATION REPORTING
# ============================================

def print_button_metrics(pred_buttons, true_buttons):
    """Print per-button validation metrics."""
    btn_names = list(BUTTON_ACTION_TOKENS)
    limit = min(len(btn_names), pred_buttons.shape[1])
    
    logger.info("-" * 95)
    logger.info(f"{'Button':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Real#':>7} {'Pred#':>7}")
    logger.info("-" * 95)
    
    macro_f1 = []
    
    for i in range(limit):
        name = btn_names[i]
        y_true = true_buttons[:, i]
        y_pred = pred_buttons[:, i]
        
        support = int(y_true.sum())
        pred_count = int(y_pred.sum())
        
        if support == 0 and pred_count == 0:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        macro_f1.append(f1)
        flag = "🔴" if rec < 0.5 and support > 10 else "🟢"
        
        logger.info(
            f"{name:<15} {acc*100:>6.1f}% {prec*100:>6.1f}% "
            f"{rec*100:>6.1f}% {f1*100:>6.1f}% "
            f"{support:>7} {pred_count:>7} {flag}"
        )
    
    avg_f1 = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0
    logger.info("-" * 95)
    logger.info(f"🏆 Macro F1 (Buttons): {avg_f1*100:.2f}%")
    logger.info("=" * 95)

# ============================================
# CUDA PREFETCHER
# ============================================

class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        
        batch = self.next_batch
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                v.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return batch

# ============================================
# DATASET
# ============================================

class GameplayDataset(Dataset):
    def __init__(self, base_path, tokenizer, image_processor, horizon=18, shift=3, transform=None):
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.horizon = horizon
        self.shift = shift
        self.base_path = Path(base_path)
        
        self.runs = {}
        self.samples = []
        self._action_indices = {}
        
        logger.info("🔍 Indexing dataset...")
        
        run_id = 0
        for run_folder in sorted(self.base_path.iterdir()):
            if not run_folder.is_dir():
                continue

            frames_dir = run_folder / "frames"
            actions_path = run_folder / "actions.jsonl"
            
            if not (frames_dir.exists() and actions_path.exists()):
                continue

            action_str = str(actions_path)
            offsets = build_line_index(action_str)
            n_frames = len(offsets)
            
            min_length = horizon + shift + 1
            if n_frames < min_length:
                continue

            self.runs[run_id] = {
                "frames_dir": str(frames_dir),
                "actions_path": action_str,
                "n_frames": n_frames,
                "emb_id": 0
            }
            self._action_indices[action_str] = offsets
            
            for i in range(0, n_frames - horizon - shift):
                self.samples.append((run_id, i))
            
            run_id += 1

        if not self.samples:
            raise ValueError("❌ Dataset empty or videos too short!")
        logger.info(f"✅ Ready: {len(self.samples):,} samples from {len(self.runs)} runs")

    def __len__(self):
        return len(self.samples)

    def _get_frame(self, frames_dir, idx):
        filename = f"frame_{idx:06d}.jpg"
        filepath = os.path.join(frames_dir, filename)
        
        img = None
        if HAS_TURBO:
            try:
                with open(filepath, 'rb') as f:
                    img = JPEG_DECODER.decode(f.read())
                img = img[:, :, ::-1].copy()
            except Exception:
                pass
        
        if img is None:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        return img

    def parse_actions(self, chunk):
        seq_len = len(chunk)
        btns_out = np.zeros((seq_len, len(BUTTON_ACTION_TOKENS)), dtype=np.float32)
        jl_out = np.zeros((seq_len, 2), dtype=np.float32)
        jr_out = np.zeros((seq_len, 2), dtype=np.float32)

        for t, entry in enumerate(chunk):
            act = entry["actions"]
            sticks = act["sticks"]
            btns = act["buttons"]
            
            jl_out[t, 0] = sticks.get("AXIS_LEFTX", 0)
            jl_out[t, 1] = sticks.get("AXIS_LEFTY", 0)
            jr_out[t, 0] = sticks.get("AXIS_RIGHTX", 0)
            jr_out[t, 1] = sticks.get("AXIS_RIGHTY", 0)
            
            for i, name in enumerate(BUTTON_ACTION_TOKENS):
                if name in btns:
                    btns_out[t, i] = float(btns[name])
                elif name in sticks:
                    val = sticks[name]
                    btns_out[t, i] = 1.0 if val > 0.5 else 0.0
                    
        return btns_out, jl_out, jr_out

    def __getitem__(self, i):
        run_id, idx = self.samples[i]
        run = self.runs[run_id]

        frame = self._get_frame(run["frames_dir"], idx)
        
        if self.transform:
            pil_img = Image.fromarray(frame)
            pil_img = self.transform(pil_img)
            frame = np.array(pil_img)
        pixel_values = self.image_processor(images=[frame], return_tensors="pt")["pixel_values"]

        actions_path = run["actions_path"]
        offsets = self._action_indices[actions_path]
        
        start_act = idx + self.shift
        chunk = []
        for j in range(self.horizon):
            action = read_json_line(actions_path, offsets[start_act + j])
            chunk.append(action)
        
        btns, jl, jr = self.parse_actions(chunk)

        raw_data = {
            "frames": pixel_values,
            "buttons": btns[None, ...],
            "j_left": jl[None, ...],
            "j_right": jr[None, ...],
            "dropped_frames": np.array([False]),
            "game": None,
            "embodiment_id": run["emb_id"]
        }

        encoded = self.tokenizer.encode(raw_data)
        
        result = {}
        for k, v in encoded.items():
            if isinstance(v, torch.Tensor):
                result[k] = v if k == "images" else v.squeeze(0)
            else:
                result[k] = v
        return result


def collate_fn(batch):
    result = {}
    first = batch[0]
    for k in first.keys():
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        elif isinstance(vals[0], np.ndarray):
            result[k] = torch.from_numpy(np.stack(vals))
        else:
            result[k] = vals
    return result


def worker_init_fn(worker_id):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(42 + worker_id)

# ============================================
# TRAIN
# ============================================

def train():
    args = parse_args()
    device = "cuda"
    
    logger.info("=" * 70)
    logger.info("🚀 NitroGen Training with EMA + Augmentations")
    logger.info("=" * 70)
    logger.info(f"Scheduler: {'OneCycle' if USE_ONECYCLE else 'Cosine+Warmup'}")
    logger.info("=" * 70)
    
    # DETERMINE CHECKPOINT PATH
    load_path = None
    is_resume = False
    
    if args.resume:
        load_path = args.resume
        is_resume = True
    elif args.resume_best:
        if os.path.exists("best_model.pt"):
            load_path = "best_model.pt"
            is_resume = True
    elif args.resume_latest:
        load_path = find_latest_checkpoint()
        if load_path:
            is_resume = True
    
    if not load_path:
        load_path = CHECKPOINT_PATH
        is_resume = False
    
    if not os.path.exists(load_path):
        logger.error(f"❌ Checkpoint not found: {load_path}")
        return
    
    logger.info(f"📂 Loading: {load_path} (resume={is_resume})")
    
    # LOAD CHECKPOINT
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    saved_ckpt_config = checkpoint["ckpt_config"]
    
    horizon = ckpt_config.model_cfg.action_horizon
    shift = ckpt_config.modality_cfg.action_shift

    # TOKENIZER & IMAGE PROCESSOR
    tokenizer = NitrogenTokenizer(ckpt_config.tokenizer_cfg)
    tokenizer.train()
    img_proc = AutoImageProcessor.from_pretrained(
        ckpt_config.model_cfg.vision_encoder_name, use_fast=True
    )
    
    logger.info(f"🖼️ Image processor: {ckpt_config.model_cfg.vision_encoder_name}")
    logger.info(f"🖼️ Target size: 256x256 (SigLIP)")

    # MODEL
    model = NitroGen(config=ckpt_config.model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    model.set_trainable_parameters(
        tune_vision_tower=False,
        tune_diffusion_model=True,
        tune_mm_projector=True,
        tune_vl_mixing=True
    )
    model.set_frozen_modules_to_eval_mode()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"🔢 Params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    ema_model.to(device)
    logger.info(f"✨ EMA decay: {EMA_DECAY}")

    # ============================================
    # DATASET
    # ============================================

    logger.info("📂 Loading dataset...")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.95, 1.0), antialias=True),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ])
    
    logger.info("🎨 Augmentations: ColorJitter only (safe for gameplay)")
    
    # Keep validation clean so checkpoint selection is not driven by augmentation noise.
    ds_train = GameplayDataset(DATASET_PATH, tokenizer, img_proc, horizon, shift, transform=train_transform)
    ds_val = GameplayDataset(DATASET_PATH, tokenizer, img_proc, horizon, shift, transform=None)

    total_samples = len(ds_train)
    val_size = int(total_samples * VAL_SPLIT)
    train_size = total_samples - val_size
    
    if train_size == 0 or val_size == 0:
        logger.error(f"❌ Dataset too small: {total_samples}")
        return
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(ds_train, train_indices)
    val_dataset = torch.utils.data.Subset(ds_val, val_indices)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "collate_fn": collate_fn,
        "pin_memory": PIN_MEMORY,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        "persistent_workers": PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        "worker_init_fn": worker_init_fn if NUM_WORKERS > 0 else None,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    
    logger.info(f"📊 Train/Val: {len(train_dataset):,} / {len(val_dataset):,}")
    logger.info(f"📊 Steps/Epoch: {steps_per_epoch:,}, Total: {total_steps:,}")

    # ============================================
    # OPTIMIZER, SCHEDULER, and RESUME STATE
    # ============================================
    base_lr = ONECYCLE_MAX_LR if USE_ONECYCLE else COSINE_LR
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=0.001,
    )
    scheduler, scheduler_meta = build_scheduler(optimizer, total_steps)
    # Store schedule metadata in checkpoints so incompatible resumes fail fast.
    training_state = make_training_state(steps_per_epoch, total_steps, scheduler_meta)
    amp_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float32

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    avg_val = float('inf')

    if is_resume:
        logger.info("🔄 Resuming training state...")

        try:
            validate_resume_compatibility(checkpoint, training_state)
        except ValueError as e:
            logger.error(f"❌ {e}")
            return

        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", start_epoch * steps_per_epoch)
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float('inf')))
        avg_val = checkpoint.get("val_loss", float('inf'))
        
        if "model_training" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_training"])
                logger.info("  ✅ Training model restored")
            except Exception as e:
                logger.warning(f"  ⚠️ Training model failed: {e}")
        
        if "ema_model" in checkpoint:
            try:
                ema_model.load_state_dict(checkpoint["ema_model"])
                logger.info("  ✅ EMA restored")
            except Exception as e:
                logger.warning(f"  ⚠️ EMA failed: {e}")
        
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("  ✅ Optimizer restored")
        except Exception as e:
            logger.error(f"  ❌ Optimizer restore failed: {e}")
            return

        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("  ✅ Scheduler restored")
        except Exception as e:
            logger.error(f"  ❌ Scheduler restore failed: {e}")
            return

        logger.info(f"  📍 Epoch {start_epoch}, Step {global_step}, Best {best_val_loss:.4f}")

    else:
        if USE_ONECYCLE:
            logger.info(f"📈 OneCycle LR: max={ONECYCLE_MAX_LR:.2e}")
        else:
            logger.info(f"📈 Cosine LR: {COSINE_LR:.2e}, warmup={scheduler_meta['warmup_steps']}")

    # ============================================
    # COMPILE
    # ============================================

    if USE_TORCH_COMPILE:
        logger.info("⚡ Compiling model...")
        model = torch.compile(model, mode=COMPILE_MODE)

    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info("🏁 Starting training loop")
    logger.info("=" * 70)

    # ============================================
    # TRAINING LOOP
    # ============================================

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        
        train_loss_sum = 0.0
        train_steps = 0
        epoch_start = time.time()

        prefetcher = CUDAPrefetcher(train_loader, device)
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for _ in pbar:
            try:
                batch = next(prefetcher)
            except StopIteration:
                break

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=amp_dtype):
                out = model(batch)
                loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("⚠️ NaN/Inf loss, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)

            global_step += 1
            train_steps += 1

            l_val = float(loss.detach().item())
            train_loss_sum += l_val

            if train_steps % 5 == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix(
                    loss=f"{l_val:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    vram=f"{vram:.1f}G"
                )

        pbar.close()

        if train_steps == 0:
            logger.warning("⚠️ No steps this epoch!")
            continue
        
        avg_train = train_loss_sum / train_steps
        epoch_time = (time.time() - epoch_start) / 60

        # ============================================
        # VALIDATION
        # ============================================

        should_validate = ((epoch + 1) % VAL_INTERVAL == 0) or ((epoch + 1) == EPOCHS)

        if should_validate:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            torch.cuda.empty_cache()
            
            val_prefetcher = CUDAPrefetcher(val_loader, device)
            check_metrics = True
            
            logger.info("🔍 Validation...")
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_prefetcher, total=len(val_loader), desc="Val", leave=False)):
                    
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        v_loss = model(batch)["loss"]
                        if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                            val_loss_sum += float(v_loss.item())
                            val_steps += 1
                    
                    if i == 0 and check_metrics:
                        try:
                            generated_dict = model.get_action(batch)
                            pred_decoded = tokenizer.decode(generated_dict)
                            pred_buttons = pred_decoded["buttons"].cpu().numpy().astype(int)
                            
                            gt_buttons = batch["buttons"].cpu().numpy().astype(int)
                            if gt_buttons.ndim == 4:
                                gt_buttons = gt_buttons.squeeze(1)
                            
                            B, T, C = gt_buttons.shape
                            
                            if pred_buttons.shape[1] != T:
                                min_t = min(T, pred_buttons.shape[1])
                                pred_buttons = pred_buttons[:, :min_t, :]
                                gt_buttons = gt_buttons[:, :min_t, :]
                            
                            pred_flat = pred_buttons.reshape(-1, C)
                            gt_flat = gt_buttons.reshape(-1, C)

                            n_steps = get_model_attr(model, 'num_inference_timesteps', '?')
                            logger.info(f"🎮 Generation (Steps: {n_steps})")
                            print_button_metrics(pred_flat, gt_flat)
                            
                            check_metrics = False
                            
                        except Exception as e:
                            logger.error(f"⚠️ Metrics failed: {e}")

            avg_val = val_loss_sum / max(val_steps, 1)
            logger.info(f"📊 Ep {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {epoch_time:.1f}m")
            torch.cuda.reset_peak_memory_stats()
            
        else:
            logger.info(f"📊 Ep {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: (skip) | Time: {epoch_time:.1f}m")

        # ============================================
        # CHECKPOINTS
        # ============================================

        ema_weights = get_ema_state_dict(ema_model)
        current_model_weights = get_model_state_dict(model)

        if should_validate and avg_val < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = avg_val
            
            best_save_dict = {
                "model": ema_weights,
                "model_training": current_model_weights,
                "ema_model": ema_model.state_dict(),
                "ckpt_config": saved_ckpt_config,
                "training_state": training_state,
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_loss": best_val_loss,
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            torch.save(best_save_dict, "best_model.pt")
            logger.info(f"🏆 New Best: {best_val_loss:.4f} (was {prev_best:.4f})")

        checkpoint_path = f"checkpoint_ep{epoch+1}.pt"
        
        torch.save({
            "model": ema_weights,
            "model_training": current_model_weights,
            "ema_model": ema_model.state_dict(),
            "ckpt_config": saved_ckpt_config,
            "training_state": training_state,
            "epoch": epoch + 1,
            "global_step": global_step,
            "val_loss": avg_val,
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, checkpoint_path)
        logger.info(f"💾 Saved: {checkpoint_path}")
        
        old_checkpoints = sorted(
            Path(".").glob("checkpoint_ep*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_ep", "") or 0)
        )
        while len(old_checkpoints) > MAX_CHECKPOINTS:
            oldest = old_checkpoints.pop(0)
            oldest.unlink()
            logger.info(f"🗑️ Removed: {oldest}")

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================
    # EXPORT
    # ============================================

    ema_weights = get_ema_state_dict(ema_model)
    
    torch.save({
        "model": ema_weights,
        "ckpt_config": saved_ckpt_config,
        "training_state": training_state,
        "epoch": EPOCHS,
        "global_step": global_step,
        "val_loss": avg_val,
        "best_val_loss": best_val_loss,
    }, "final_model.pt")
    
    logger.info("=" * 70)
    logger.info("💾 Saved: final_model.pt (EMA weights)")
    logger.info("=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    train()
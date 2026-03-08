import time
import torch
import numpy as np
from transformers import AutoImageProcessor

from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer
from nitrogen.cfg import CkptConfig
from nitrogen.shared import PATH_REPO

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def summarize_parameters(module, name='model', depth=0, max_depth=3):
    """
    Print a tree-like summary of parameters in a PyTorch module.
    
    Args:
        module: PyTorch module to summarize
        name: Name of the module (for root level)
        depth: Current depth in the tree
        max_depth: Maximum depth to traverse
    """
    if depth > max_depth:
        return
    
    # Count total parameters in this module
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print indented summary
    indent = "  " * depth
    print(f"{indent}{name}: {total_params:,} params ({trainable_params:,} trainable)")
    
    # Recursively summarize submodules
    if depth < max_depth:
        for child_name, child_module in module.named_children():
            summarize_parameters(child_module, child_name, depth + 1, max_depth)


def load_model(checkpoint_path: str):
    """Load model and args from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg = ckpt_config.model_cfg
    tokenizer_cfg = ckpt_config.tokenizer_cfg

    print("\nCheckpoint config:")
    print(f"  Model type: {model_cfg.model_type}")
    print(f"  Action dim: {model_cfg.action_dim}")
    print(f"  Action horizon: {model_cfg.action_horizon}")
    print(f"  Vision encoder: {model_cfg.vision_encoder_name}")

    # Initialize tokenizer and language model
    img_proc = AutoImageProcessor.from_pretrained(
        model_cfg.vision_encoder_name,
        use_fast=True
    )

    # Create VLM with pre-loaded language model
    if isinstance(model_cfg, NitroGen_Config):
        assert isinstance(tokenizer_cfg, NitrogenTokenizerConfig)
        
        tokenizer_cfg.training = False
        if tokenizer_cfg.game_mapping_cfg is not None:
            tokenizer_cfg.game_mapping_cfg.src_files = [
                x.replace("/mnt/amlfs-02/shared/gaming/gamingvla", str(PATH_REPO))
                for x in tokenizer_cfg.game_mapping_cfg.src_files
            ]
        
        tokenizer = NitrogenTokenizer(tokenizer_cfg)
        game_mapping = tokenizer.game_mapping
        model = NitroGen(config=model_cfg, game_mapping=game_mapping)
    else:
        raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

    print("\nModel architecture:")
    summarize_parameters(model, max_depth=2)

    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    tokenizer.eval()
    
    model = model.cuda().float()
    
    print(f"\n✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    return model, tokenizer, img_proc, ckpt_config, game_mapping


class InferenceSession:
    """Manages state for a single inference session."""
    
    def __init__(
        self,
        model: NitroGen,
        tokenizer: NitrogenTokenizer,
        img_proc,
        ckpt_config: CkptConfig,
        game_mapping: dict,
        selected_game: str,
        old_layout: bool = False,
        cfg_scale: float = 1.0,
        actions_per_step: int = None,
        num_inference_steps: int = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.img_proc = img_proc
        self.ckpt_config = ckpt_config
        self.game_mapping = game_mapping
        self.selected_game = selected_game
        self.old_layout = old_layout
        self.cfg_scale = cfg_scale
        self.actions_per_step = actions_per_step
        # Single-frame runtime keeps per-step tensors on device.
        self.device = "cuda"
        self.dtype = torch.float32
        
        if num_inference_steps is not None:
            self.model.num_inference_timesteps = num_inference_steps
        
        self._compiled = False
        self._setup_buffers()
        
    def _setup_buffers(self):
        """Pre-allocate tensors."""
        # Fixed IDs used by tokenizer/model at inference time.
        self._frame_buffer = None
        self._embodiment_id = torch.tensor([0], dtype=torch.long, device=self.device)
        
        game_id = 0
        if self.game_mapping and self.selected_game:
            game_id = self.game_mapping.get(self.selected_game, 0)
        self._game_ids = torch.tensor([game_id], dtype=torch.long, device=self.device)
        self._game_ids_uncond = torch.tensor([0], dtype=torch.long, device=self.device)

    def compile(self):
        """Compile model with torch.compile."""
        if self._compiled:
            print("  Model already compiled")
            return
            
        print("\n" + "="*60)
        print("COMPILING MODEL")
        print("="*60)
        
        try:
            self.model.get_action = torch.compile(
                self.model.get_action, 
                mode="default" 
            )
            print("  ✓ Compiled: get_action")
            
            if self.cfg_scale != 1.0:
                self.model.get_action_with_cfg = torch.compile(
                    self.model.get_action_with_cfg, 
                    mode="default"
                )
                print("  ✓ Compiled: get_action_with_cfg")
            
            self._compiled = True
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")
        
        print("="*60 + "\n")
    
    def warmup(self, iterations: int = 5):
        """Warmup the model."""
        print(f"\nWarmup: {iterations} iterations...")
        
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        times = []
        for i in range(iterations):
            start = time.time()
            _ = self.predict(dummy_img)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.1f}ms")
        
        if len(times) > 1:
            speedup = times[0] / times[-1]
            print(f"  Speedup: {speedup:.1f}x (first: {times[0]:.1f}ms, last: {times[-1]:.1f}ms)")
        print("✓ Warmup complete\n")
    
    def reset(self):
        """Reset session state."""
        return None
    
    def predict(self, obs, profile: bool = False) -> dict:
        """
        Predict actions from single observation.
        
        Args:
            obs: RGB image as numpy array (H, W, 3)
            profile: Enable profiling
            
        Returns:
            dict with j_left, j_right, buttons
        """
        # 1. Image preprocessing
        pixel_values = self.img_proc([obs], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=self.dtype, non_blocking=True)
        
        # 2. Single frame - no history needed
        frame = pixel_values # Shape: (1, C, H, W)
        dropped_frames = torch.zeros(1, dtype=torch.bool, device=self.device)
        
        # 3. Tokenize
        data_cond = {
            "frames": frame.squeeze(0), # Remove batch dim for tokenizer
            "dropped_frames": dropped_frames,
            "game": self.selected_game,
        }
        
        tokenized_cond = self.tokenizer.encode(data_cond)
        
        for k, v in tokenized_cond.items():
            if isinstance(v, torch.Tensor):
                v = v.unsqueeze(0).to(self.device, non_blocking=True)
                if k == "images" and v.ndim == 4:
                    v = v.unsqueeze(1)
                tokenized_cond[k] = v
            elif isinstance(v, np.ndarray):
                tokenized_cond[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
            else:
                tokenized_cond[k] = [v]
        
        tokenized_cond["embodiment_id"] = self._embodiment_id
        tokenized_cond["game_ids"] = self._game_ids
        
        # 4. CFG (if enabled)
        tokenized_uncond = None
        if self.cfg_scale != 1.0:
            dropped_frames_uncond = torch.ones(1, dtype=torch.bool, device=self.device)
            
            data_uncond = {
                "frames": frame.squeeze(0),
                "dropped_frames": dropped_frames_uncond,
                "game": None
            }
            tokenized_uncond = self.tokenizer.encode(data_uncond)
            
            # Convert to CUDA tensors with batch dimension
            for k, v in tokenized_uncond.items():
                if isinstance(v, torch.Tensor):
                    tokenized_uncond[k] = v.unsqueeze(0).to(self.device, non_blocking=True)
                elif isinstance(v, np.ndarray):
                    tokenized_uncond[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                else:
                    tokenized_uncond[k] = [v]
            
            tokenized_uncond["embodiment_id"] = self._embodiment_id
            tokenized_uncond["game_ids"] = self._game_ids_uncond
        
        # 5. Inference
        with torch.inference_mode():
            torch.compiler.cudagraph_mark_step_begin()
            
            if self.cfg_scale == 1.0:
                model_output = self.model.get_action(
                    tokenized_cond, 
                    old_layout=self.old_layout, 
                    profile=profile
                )
            else:
                model_output = self.model.get_action_with_cfg(
                    tokenized_cond, 
                    tokenized_uncond, 
                    cfg_scale=self.cfg_scale
                )
        
        timings = model_output.get("_timings")
        predicted = self.tokenizer.decode(model_output)
        
        # 6. Extract actions
        j_left = predicted["j_left"].squeeze().cpu().numpy()
        j_right = predicted["j_right"].squeeze().cpu().numpy()
        buttons = predicted["buttons"].squeeze().cpu().numpy()
        
        # 7. Receding horizon (if enabled)
        if self.actions_per_step is not None:
            n = self.actions_per_step
            if j_left.ndim > 0 and len(j_left) > n:
                j_left = j_left[:n]
                j_right = j_right[:n]
                buttons = buttons[:n]
        
        return {
            "j_left": j_left, 
            "j_right": j_right, 
            "buttons": buttons, 
            "timings": timings
        }
    
    def info(self) -> dict:
        """Get session information."""
        return {
            "selected_game": self.selected_game,
            "old_layout": self.old_layout,
            "cfg_scale": self.cfg_scale,
            "actions_per_step": self.actions_per_step,
            "num_inference_steps": self.model.num_inference_timesteps,
            "compiled": self._compiled,
            "mode": "single_frame",
        }
    
    @classmethod
    def from_ckpt(
        cls,
        checkpoint_path: str,
        old_layout: bool = False,
        cfg_scale: float = 1.0,
        compile_model: bool = True,
        actions_per_step: int = None,
        num_inference_steps: int = None,
    ):
        """
        Create InferenceSession from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            old_layout: Use old action layout
            cfg_scale: Classifier-free guidance scale (1.0 = disabled)
            compile_model: Use torch.compile
            actions_per_step: Use only first N actions (Receding Horizon)
            num_inference_steps: Override flow matching steps
        
        Returns:
            InferenceSession
        """
        model, tokenizer, img_proc, ckpt_config, game_mapping = load_model(checkpoint_path)
        
        selected_game = None
        if game_mapping:
            print("\n" + "="*60)
            print("GAME SELECTION")
            print("="*60)
            print("Available games:")
            for game, idx in sorted(game_mapping.items(), key=lambda x: x[1]):
                if game is not None:
                    print(f"  {idx:03d}: {game}")
            print("  (empty): Unconditional generation")
            print("-"*60)
            
            choice = input("Enter game ID or name: ").strip()
            
            if choice == "":
                selected_game = None
                print("✓ Using unconditional generation")
            else:
                try:
                    idx = int(choice)
                    candidates = [k for k, v in game_mapping.items() if v == idx]
                    if candidates:
                        selected_game = candidates[0]
                        print(f"✓ Selected: {selected_game}")
                    else:
                        print(f"⚠ No game with ID {idx}, using unconditional")
                except ValueError:
                    if choice in game_mapping:
                        selected_game = choice
                        print(f"✓ Selected: {selected_game}")
                    else:
                        print(f"⚠ Game '{choice}' not found, using unconditional")
            print("="*60)
        
        session = cls(
            model=model,
            tokenizer=tokenizer,
            img_proc=img_proc,
            ckpt_config=ckpt_config,
            game_mapping=game_mapping,
            selected_game=selected_game,
            old_layout=old_layout,
            cfg_scale=cfg_scale,
            actions_per_step=actions_per_step,
            num_inference_steps=num_inference_steps,
        )
        
        if compile_model:
            session.compile()
        
        return session
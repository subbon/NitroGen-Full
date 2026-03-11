<img src="assets/github_banner.gif" width="100%" />

# **NitroGen Full** — Fine-Tuning Fork

This is a practical fork of the original **NitroGen** project focused on fine-tuning NitroGen for **any game** from locally recorded gameplay. The core idea stays the same as upstream NitroGen: predict controller actions directly from pixels. This fork adds a complete adaptation workflow around a custom game-specific dataset, direct local inference, and utilities for training and validation.

This cleaned release folder keeps only the NitroGen-related path documented here:

- **`nitrogen/`**
- **`play.py`**
- **`train.py`**
- **`gui_recorder.py`**

## What Changed Compared to Original NitroGen

The original project is a minimal research release: install the package, launch a **ZeroMQ** inference server with `scripts/serve.py`, and connect to it from `scripts/play.py`.

This fork changes that workflow in several practical ways:

1. **Direct local inference** instead of server/client split  
   Upstream NitroGen uses `serve.py` + `play.py` with ZeroMQ. This fork loads the checkpoint directly inside **[`play.py`](./play.py)**, so there is no separate inference server to start.

2. **A dedicated fine-tuning pipeline**  
   **[`train.py`](./train.py)** adds a full post-training workflow on a custom gameplay dataset: **EMA**, **BF16/AMP**, **`torch.compile`**, validation, checkpoint selection, and button-level metrics.

3. **A custom data collection path**  
   **[`gui_recorder.py`](./gui_recorder.py)** records gameplay frames and synchronized controller actions into a dataset layout that can be consumed by the fine-tuning script.

4. **Training changes for rare and active actions**  
   This fork changes the training objective so active action dimensions receive a stronger loss weight. In the current model code, action targets with magnitude above **`0.1`** are upweighted by **`x3.5`**, which is intended to make rarer button presses and non-neutral actions matter more during fine-tuning.

5. **Game-agnostic adaptation workflow**  
   Upstream NitroGen is presented as a generalist gaming foundation model. This fork is organized around adapting the base checkpoint to **any** target game through custom gameplay recordings.

## Current Training Snapshot

### Latest Training (1.0x weights — `final_model.pt`)

- **Base checkpoint**: **`ng.pt`**
- **Dataset size**: **34 runs**, **681,203 samples**
- **Train/val split**: **613,083 / 68,120**
- **Trainable parameters**: **177,675,289 / 493,631,513** (36.0%)
- **Training environment**: **WSL + RTX 4070 Ti Super** (~13GB VRAM usage)
- **Schedule**: **OneCycle**
- **Total epochs**: **10**
- **Best validation loss**: **`0.0160`** (EMA checkpoint, epoch 10)
- **Best Macro F1**: **44.61%** (per benchmark)
- **Date**: **March 9, 2026**

### Previous Training (3.5x weights — `final_model_35.pt`)

- **Loss Weight**: **3.5x** for active actions
- **Best validation loss**: **`0.0209`**
- **Best Macro F1**: **45.49%** (per benchmark)
- **Date**: **March 6, 2026**

### Benchmark Results

**Key findings** (benchmarked on validation set, 200 batches, 2 runs per model):
- Fine-tuned models show massive improvement over base (**7.24% → 44-45% Macro F1**)
- **`final_model.pt`** achieves best loss (**0.0158**) with competitive F1 (**44.61%**) — recommended for gameplay
- **`final_model_35.pt`** achieves best **Macro F1: 45.49%** but higher loss due to aggressive upweighting
- 3.5x model has higher recall on rare actions but may be less predictable in practice

## Available Models

| Model | Loss Weight | Loss | Macro F1 | Description |
|-------|-------------|------|----------|-------------|
| **`final_model.pt`** | **1.0x** | **0.0158** | **44.61%** | **Recommended** — Standard loss weights, best stability for gameplay |
| **`final_model_35.pt`** | **3.5x** | **0.0209** | **45.49%** | **Best Macro F1** — upweighted actions, potentially less stable |

**Download from [HuggingFace](https://huggingface.co/subbonan/nitrogen-pizza-tower-finetune)**

**Recommendation:** Use **`final_model.pt`** (1.0x) for actual gameplay — it provides more consistent behavior despite slightly lower F1 score. The 3.5x model upweights rare actions heavily which can lead to unpredictable button spam.

## Gameplay Examples

### 1.0x Model (`final_model.pt`) — Recommended

<img src="assets/10_banner.gif" width="100%" />

### 3.5x Model (`final_model_35.pt`)

<img src="assets/35_banner.gif" width="100%" />

## Project Layout

```text
repo/
|-- nitrogen/                  # Modified NitroGen package used by this fork
|-- play.py                    # Direct local inference on Windows
|-- train.py                   # NitroGen fine-tuning script
|-- gui_recorder.py            # Gameplay recorder for custom dataset creation
```

**Key components:**
- **`nitrogen/`** — Modified NitroGen package used by this fork
- **`play.py`** — Direct local inference on Windows
- **`train.py`** — NitroGen fine-tuning script
- **`gui_recorder.py`** — Gameplay recorder for custom dataset creation

## Installation

Clone and install the base package:

```bash
git clone https://github.com/subbon/NitroGen-Full.git
pip install -e .
```

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| **`[train]`** | bitsandbytes, scikit-learn, torchvision, tqdm, PyTurboJPEG | Fine-tuning the model |
| **`[play]`** | dxcam, vgamepad, pygame, keyboard, pywinctl, pywin32, xspeedhack | Gameplay inference and recording on Windows |
| **`[train,play]`** | All of the above | Full setup for training + inference |

**Training only (Linux/WSL):**
```bash
pip install -e .[train]
```

**Inference/recording only (Windows):**
```bash
pip install -e .[play]
```

**Full setup (training + inference):**
```bash
pip install -e .[train,play]
```

**Note:** `[play]` includes Windows-specific packages that will only install on Windows (`sys_platform == 'win32'`). On Linux these are skipped automatically.

### Manual Package Install (if not using optional deps)

Additional packages for training and recording:
```bash
pip install bitsandbytes scikit-learn torchvision tqdm pygame keyboard PyTurboJPEG
```

### Requirements

- **Training / fine-tuning**: **CUDA Linux or WSL**
- **Gameplay inference / recording**: **Windows** (requires gamepad and capture libraries)
- **GPU**: **CUDA-capable GPU** required for both training and inference
- **Game**: Your own legal copy of the target game

## Dataset Format

The fine-tuning script expects a dataset directory with one folder per recorded run:

```text
~/my_dataset/
|-- run_20260101_120000/
|   |-- frames/
|   |   |-- frame_000000.jpg
|   |   |-- frame_000001.jpg
|   |   \-- ...
|   \-- actions.jsonl
|-- run_20260101_123000/
|   |-- frames/
|   \-- actions.jsonl
\-- ...
```

Each line in **`actions.jsonl`** follows the structure produced by **[`gui_recorder.py`](./gui_recorder.py)**:

```json
{
  "frame": 0,
  "timestamp": 0.016667,
  "sync_diff_ms": 1.24,
  "actions": {
    "buttons": {
      "SOUTH": 0,
      "EAST": 0,
      "WEST": 1,
      "NORTH": 0,
      "LEFT_SHOULDER": 0,
      "RIGHT_SHOULDER": 0,
      "BACK": 0,
      "START": 0,
      "LEFT_THUMB": 0,
      "RIGHT_THUMB": 0,
      "GUIDE": 0,
      "DPAD_UP": 0,
      "DPAD_DOWN": 0,
      "DPAD_LEFT": 1,
      "DPAD_RIGHT": 0
    },
    "sticks": {
      "AXIS_LEFTX": -1.0,
      "AXIS_LEFTY": 0.0,
      "AXIS_RIGHTX": 0.0,
      "AXIS_RIGHTY": 0.0,
      "LEFT_TRIGGER": 0.0,
      "RIGHT_TRIGGER": 1.0
    }
  }
}
```

## Recording a Dataset

Use **[`gui_recorder.py`](./gui_recorder.py)** on Windows to build the dataset:

```bash
python gui_recorder.py
```

**Workflow:**

1. Select the game window.
2. Select the controller.
3. Press **`F9`** to start recording.
4. Press **`F9`** again to stop.
5. Move or rename the recorded runs into the dataset root expected by training, for example **`~/my_dataset`**.

The recorder writes:

- **JPEG frames** to **`frames/frame_XXXXXX.jpg`**
- **Synchronized controller events** to **`actions.jsonl`**

## Fine-Tuning NitroGen

The main training entry point is **[`train.py`](./train.py)**.

Before launching training, edit the constants near the top of the file:

- **`DATASET_PATH`**
- **`CHECKPOINT_PATH`**
- **`BATCH_SIZE`**
- **`EPOCHS`**
- Scheduler settings such as **`USE_ONECYCLE`**, **`ONECYCLE_MAX_LR`**, or cosine settings

**Basic run:**

```bash
python train.py
```

**Resume options** exposed by the script:

```bash
python train.py --resume /path/to/checkpoint.pt
python train.py --resume-best
python train.py --resume-latest
```

**Important caveat:**

- Resume is supported only when the checkpoint contains full optimizer/scheduler state and matching training metadata.
- For **`OneCycle`**, resume is allowed only if the current schedule matches the saved checkpoint schedule exactly.

**Default training behavior** in the current script:

- Loads weights from checkpoint (**`ng.pt`** for fine-tuning start, or **`final_model_35.pt`** for inference)
- **Freezes the vision tower**
- Trains the **diffusion model**, **multimodal projector**, and **VL mixing blocks**
- Upweights active action targets by **`x3.5`** in the flow-matching loss to emphasize rarer button presses and non-neutral control states
- Uses **EMA** with decay **`0.9999`**
- Uses **BF16 AMP** when enabled
- Uses **`torch.compile`**
- Logs button-level validation metrics

**Generated outputs:**

- **`final_model.pt`** — Standard 1.0x weights (recommended)
- **`final_model_35.pt`** — 3.5x upweighted version
- **`training_wsl.log`**
- **`checkpoint_ep*.pt`**

The script keeps rolling epoch checkpoints named **`checkpoint_ep*.pt`** and prunes old ones according to **`MAX_CHECKPOINTS`**.

## Inference and Gameplay

This fork does not require the original **`scripts/serve.py`** inference server. The checkpoint is loaded directly by **[`play.py`](./play.py)**.

**Recommended (1.0x model — more stable):**
```bash
python play.py path\to\final_model.pt --process PizzaTower.exe
```

**Alternative (3.5x model — higher F1 but less stable):**
```bash
python play.py path\to\final_model_35.pt --process PizzaTower.exe
```

**Note:** If you get **`Error 5: Access Denied`** when initializing speedhack, run **CMD or PowerShell as Administrator**, then execute the command from there. This is only needed if the speedhack injection fails.

**Useful options:**

```bash
python play.py path\to\final_model.pt --process PizzaTower.exe --cfg 1.0
python play.py path\to\final_model.pt --process PizzaTower.exe --steps 16
python play.py path\to\final_model.pt --process PizzaTower.exe --actions-per-step 4
python play.py path\to\final_model.pt --process PizzaTower.exe --no-record
python play.py path\to\final_model.pt --process PizzaTower.exe --no-debug
python play.py path\to\final_model.pt --process PizzaTower.exe --no-compile
python play.py path\to\final_model.pt --process PizzaTower.exe --no-warmup
```

**What `play.py` does:**

- Loads the **NitroGen checkpoint** directly
- Optionally **compiles** the action generation path
- **Warms up** the model
- **Captures** the game window
- **Predicts** controller actions
- **Sends** those actions through a virtual gamepad
- Optionally **records** clean and debug videos

**Outputs are written to:**

- **`out/<checkpoint_name>/`**
- **`debug/`**

## Utility Scripts

Useful support script in this folder:

- **[`gui_recorder.py`](./gui_recorder.py)**: records new gameplay data in the expected format.

## Files to Exclude

When publishing or sharing this project, exclude:

- **`debug/`**
- **`out/`**
- **`screenshots/`**
- **`logs/`**
- **`__pycache__/`**

## Attribution

This work is based on the original **NitroGen** project by the NitroGen authors. If you publish this fork or its weights, keep the original attribution, paper link, and license notice.

**Original project:**

- **NitroGen repository**: https://github.com/MineDojo/NitroGen
- **Paper**: https://arxiv.org/abs/2601.02427

## Citation

If you release this publicly, cite the original **NitroGen** paper and optionally add your own fork-specific citation entry once you have a stable release name.

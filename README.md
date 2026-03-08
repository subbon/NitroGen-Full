<img src="assets/github_banner.gif" width="100%" />

# NitroGen Full — Fine-Tuning Fork

This is a practical fork of the original NitroGen project focused on fine-tuning NitroGen for **any game** from locally recorded gameplay. The core idea stays the same as upstream NitroGen: predict controller actions directly from pixels. This fork adds a complete adaptation workflow around a custom game-specific dataset, direct local inference, and utilities for training and validation.

This cleaned release folder keeps only the NitroGen-related path documented here:

- `nitrogen/`
- `play.py`
- `train.py`
- `gui_recorder.py`

## What Changed Compared to Original NitroGen

The original project is a minimal research release: install the package, launch a ZeroMQ inference server with `scripts/serve.py`, and connect to it from `scripts/play.py`.

This fork changes that workflow in several practical ways:

1. Direct local inference instead of server/client split  
   Upstream NitroGen uses `serve.py` + `play.py` with ZeroMQ. This fork loads the checkpoint directly inside [`play.py`](./play.py), so there is no separate inference server to start.

2. A dedicated fine-tuning pipeline  
   [`train.py`](./train.py) adds a full post-training workflow on a custom gameplay dataset: EMA, BF16/AMP, `torch.compile`, validation, checkpoint selection, and button-level metrics.

3. A custom data collection path  
   [`gui_recorder.py`](./gui_recorder.py) records gameplay frames and synchronized controller actions into a dataset layout that can be consumed by the fine-tuning script.

4. Training changes for rare and active actions  
   This fork changes the training objective so active action dimensions receive a stronger loss weight. In the current model code, action targets with magnitude above `0.1` are upweighted by `x3.5`, which is intended to make rarer button presses and non-neutral actions matter more during fine-tuning.

5. Game-agnostic adaptation workflow  
   Upstream NitroGen is presented as a generalist gaming foundation model. This fork is organized around adapting the base checkpoint to **any** target game through custom gameplay recordings.

## Current Training Snapshot

The latest training:

- base checkpoint: `ng.pt` (original NitroGen foundation model, compatible with this fork)
- dataset size: 34 runs, 681,203 samples
- train/val split: 613,083 / 68,120
- trainable parameters: 177,675,289 / 493,631,513 (36.0%)
- training environment: WSL + RTX 4070 Ti Super (~13GB VRAM usage with current settings)
- schedule: OneCycle
- total epochs: 10
- best validation loss: `0.0314` (EMA checkpoint, epoch 10)
- final log entry date: March 6, 2026

The training script saves:

- `final_model_35.pt` for the last EMA checkpoint (3.5x loss weight)

## Available Models

| Model | Loss Weight | Description |
|-------|-------------|-------------|
| `final_model_35.pt` | 3.5x | Current model. Upweighted active actions (3.5x). Higher validation metrics but potentially more unpredictable in-game behavior. |
| `final_model.pt` | 1.0x | *Coming soon on [HuggingFace](https://huggingface.co/subbonan/nitrogen-pizza-tower-finetune).* Standard loss weights for more stable gameplay. |

Download from [HuggingFace](https://huggingface.co/subbonan/nitrogen-pizza-tower-finetune)

Gameplay example with 3.5x model:

<img src="assets/35_banner.gif" width="100%" />

Note on 3.5x weighting: The 3.5x upweighting improves validation metrics (loss, F1) by emphasizing rare button presses. However, based on gameplay observations, this may make the model less predictable in practice. The upcoming 1.0x model should provide more consistent behavior.

## Project Layout

```text
repo/
|-- nitrogen/                  # Modified NitroGen package used by this fork
|-- play.py                    # Direct local inference on Windows
|-- train.py                   # NitroGen fine-tuning script
|-- gui_recorder.py            # Gameplay recorder for custom dataset creation
```

## Installation

Clone and install the base package:

```bash
git clone https://github.com/subbon/NitroGen-Full.git
pip install -e .
```

### Optional dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[train]` | bitsandbytes, scikit-learn, torchvision, tqdm, PyTurboJPEG | Fine-tuning the model |
| `[play]` | dxcam, vgamepad, pygame, keyboard, pywinctl, pywin32, xspeedhack | Gameplay inference and recording on Windows |
| `[train,play]` | All of the above | Full setup for training + inference |

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

### Manual package install (if not using optional deps)

Additional packages for training and recording:
```bash
pip install bitsandbytes scikit-learn torchvision tqdm pygame keyboard PyTurboJPEG
```

### Requirements

- **Training / fine-tuning**: CUDA Linux or WSL
- **Gameplay inference / recording**: Windows (requires gamepad and capture libraries)
- **GPU**: CUDA-capable GPU required for both training and inference
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

Each line in `actions.jsonl` follows the structure produced by [`gui_recorder.py`](./gui_recorder.py):

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

Use [`gui_recorder.py`](./gui_recorder.py) on Windows to build the dataset:

```bash
python gui_recorder.py
```

Workflow:

1. Select the game window.
2. Select the controller.
3. Press `F9` to start recording.
4. Press `F9` again to stop.
5. Move or rename the recorded runs into the dataset root expected by training, for example `~/my_dataset`.

The recorder writes:

- JPEG frames to `frames/frame_XXXXXX.jpg`
- synchronized controller events to `actions.jsonl`

## Fine-Tuning NitroGen

The main training entry point is [`train.py`](./train.py).

Before launching training, edit the constants near the top of the file:

- `DATASET_PATH`
- `CHECKPOINT_PATH`
- `BATCH_SIZE`
- `EPOCHS`
- scheduler settings such as `USE_ONECYCLE`, `ONECYCLE_MAX_LR`, or cosine settings

Basic run:

```bash
python train.py
```

Resume options exposed by the script:

```bash
python train.py --resume /path/to/checkpoint.pt
python train.py --resume-best
python train.py --resume-latest
```

Important caveat:

- Resume is supported only when the checkpoint contains full optimizer/scheduler state and matching training metadata.
- For `OneCycle`, resume is allowed only if the current schedule matches the saved checkpoint schedule exactly.

Default training behavior in the current script:

- loads weights from checkpoint (`ng.pt` for fine-tuning start, or `final_model_35.pt` for inference)
- freezes the vision tower
- trains the diffusion model, multimodal projector, and VL mixing blocks
- upweights active action targets by `x3.5` in the flow-matching loss to emphasize rarer button presses and non-neutral control states
- uses EMA with decay `0.9999`
- uses BF16 AMP when enabled
- uses `torch.compile`
- logs button-level validation metrics

Generated outputs:

- `final_model_35.pt`
- `training_wsl.log`
- `checkpoint_ep*.pt`

The script keeps rolling epoch checkpoints named `checkpoint_ep*.pt` and prunes old ones according to `MAX_CHECKPOINTS`.

## Inference and Gameplay

This fork does not require the original `scripts/serve.py` inference server. The checkpoint is loaded directly by [`play.py`](./play.py).

Example:

```bash
python play.py path\to\final_model_35.pt --process PizzaTower.exe
```

**Note:** If you get `Error 5: Access Denied` when initializing speedhack, run **CMD or PowerShell as Administrator**, then execute the command from there. This is only needed if the speedhack injection fails.

Useful options:

```bash
python play.py path\to\final_model_35.pt --process PizzaTower.exe --cfg 1.0
python play.py path\to\final_model_35.pt --process PizzaTower.exe --steps 16
python play.py path\to\final_model_35.pt --process PizzaTower.exe --actions-per-step 4
python play.py path\to\final_model_35.pt --process PizzaTower.exe --no-record
python play.py path\to\final_model_35.pt --process PizzaTower.exe --no-debug
python play.py path\to\final_model_35.pt --process PizzaTower.exe --no-compile
python play.py path\to\final_model_35.pt --process PizzaTower.exe --no-warmup
```

What `play.py` does:

- loads the NitroGen checkpoint directly
- optionally compiles the action generation path
- warms up the model
- captures the game window
- predicts controller actions
- sends those actions through a virtual gamepad
- optionally records clean and debug videos

Outputs are written to:

- `out/<checkpoint_name>/`
- `debug/`

## Utility Scripts

Useful support script in this folder:

- [`gui_recorder.py`](./gui_recorder.py): records new gameplay data in the expected format.

## Files to Exclude

When publishing or sharing this project, exclude:

- `debug/`
- `out/`
- `screenshots/`
- `logs/`
- `__pycache__/`

## Attribution

This work is based on the original NitroGen project by the NitroGen authors. If you publish this fork or its weights, keep the original attribution, paper link, and license notice.

Original project:

- NitroGen repository: https://github.com/MineDojo/NitroGen
- Paper: https://arxiv.org/abs/2601.02427

## Citation

If you release this publicly, cite the original NitroGen paper and optionally add your own fork-specific citation entry once you have a stable release name.
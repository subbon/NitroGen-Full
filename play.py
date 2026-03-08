import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_session import InferenceSession

# ============================================
# ARGS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="NitroGen Inference (Windows)")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--old-layout", action="store_true", 
                        help="Use old layout (for legacy checkpoints)")
    parser.add_argument("--cfg", type=float, default=1.0, 
                        help="CFG scale (1.0 = no CFG)")
    parser.add_argument("--steps", type=int, default=None, 
                        help="Number of inference steps (fewer = faster)")
    parser.add_argument("--actions-per-step", type=int, default=None,
                        help="Use only first N actions per prediction")
    
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-warmup", action="store_true", 
                        help="Skip warmup iterations")
    
    parser.add_argument("--process", type=str, default="celeste.exe", 
                        help="Game process name")
    parser.add_argument("--allow-menu", action="store_true", 
                        help="Allow menu buttons (START, GUIDE, BACK)")
    
    parser.add_argument("--no-record", action="store_true", 
                        help="Disable video recording")
    parser.add_argument("--no-debug", action="store_true", 
                        help="Don't save debug PNG frames")
    
    return parser.parse_args()

# ============================================
# ACTION HELPERS
# ============================================

def create_zero_action():
    """Create a zero/neutral gamepad action."""
    return OrderedDict([
        ("WEST", 0),
        ("SOUTH", 0),
        ("BACK", 0),
        ("DPAD_DOWN", 0),
        ("DPAD_LEFT", 0),
        ("DPAD_RIGHT", 0),
        ("DPAD_UP", 0),
        ("GUIDE", 0),
        ("AXIS_LEFTX", np.array([0], dtype=np.int64)),
        ("AXIS_LEFTY", np.array([0], dtype=np.int64)),
        ("LEFT_SHOULDER", 0),
        ("LEFT_TRIGGER", np.array([0], dtype=np.int64)),
        ("AXIS_RIGHTX", np.array([0], dtype=np.int64)),
        ("AXIS_RIGHTY", np.array([0], dtype=np.int64)),
        ("LEFT_THUMB", 0),
        ("RIGHT_THUMB", 0),
        ("RIGHT_SHOULDER", 0),
        ("RIGHT_TRIGGER", np.array([0], dtype=np.int64)),
        ("START", 0),
        ("EAST", 0),
        ("NORTH", 0),
    ])

# ============================================
# ENVIRONMENT BOOTSTRAP
# ============================================

def setup_game_env(process_name: str):
    """Setup game environment."""
    env = GamepadEnv(
        game=process_name,
        game_speed=1.0,
        env_fps=60,
        async_mode=True,
    )
    
    if process_name in ["isaac-ng.exe", "Cuphead.exe"]:
        print(f"\nGamepadEnv ready for {process_name}")
        input("Press Enter to initialize controller...")
        
        def press(button):
            env.gamepad_emulator.press_button(button)
            env.gamepad_emulator.gamepad.update()
            time.sleep(0.05)
            env.gamepad_emulator.release_button(button)
            env.gamepad_emulator.gamepad.update()
        
        print("Initializing controller...")
        press("SOUTH")
        for _ in range(5):
            press("EAST")
            time.sleep(0.3)
    
    return env

# ============================================
# MAIN LOOP
# ============================================

def main():
    args = parse_args()
    CKPT_NAME = Path(args.ckpt).stem
    NO_MENU = not args.allow_menu
    
    PATH_DEBUG = PATH_REPO / "debug"
    if not args.no_debug:
        PATH_DEBUG.mkdir(parents=True, exist_ok=True)
    
    PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
    PATH_OUT.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
    next_number = 1
    if video_files:
        existing_numbers = [f.name.split("_")[0] for f in video_files if f.name.split("_")[0].isdigit()]
        if existing_numbers:
            next_number = max(int(n) for n in existing_numbers) + 1
    
    PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
    PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
    PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.jsonl"
    
    BUTTON_PRESS_THRES = 0.5
    TOKEN_SET = BUTTON_ACTION_TOKENS
    
    print("\n" + "="*80)
    print("NITROGEN INFERENCE")
    print("="*80)
    
    session = InferenceSession.from_ckpt(
        checkpoint_path=args.ckpt,
        old_layout=args.old_layout,
        cfg_scale=args.cfg,
        compile_model=not args.no_compile,
        actions_per_step=args.actions_per_step,
        num_inference_steps=args.steps,
    )
    
    if not args.no_warmup:
        session.warmup(iterations=5)
    
    policy_info = session.info()
    
    print("\n" + "="*60)
    print("SESSION CONFIGURATION")
    print("="*60)
    for k, v in sorted(policy_info.items()):
        print(f"  {k:25s}: {v}")
    print("="*60)
    
    zero_action = create_zero_action()
    
    print(f"\nStarting {args.process} in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    env = setup_game_env(args.process)
    env.reset()
    env.pause()
    
    obs, reward, terminated, truncated, info = env.step(action=zero_action)
    
    step_count = 0
    inference_times = []
    
    debug_recorder = None
    clean_recorder = None
    
    if not args.no_record:
        print(f"Recording enabled:")
        print(f"  Debug: {PATH_MP4_DEBUG}")
        print(f"  Clean: {PATH_MP4_CLEAN}")
        debug_recorder = VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="fast")
        clean_recorder = VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="fast")
    
    print("\n" + "="*80)
    print("STARTING GAME LOOP (Ctrl+C to stop)")
    print("="*80 + "\n")
    
    try:
        while True:
            if not args.no_debug:
                obs.save(PATH_DEBUG / f"{step_count:05d}.png")
            
            infer_start = time.time()
            pred = session.predict(obs, profile=True)
            infer_time = time.time() - infer_start
            inference_times.append(infer_time)
            
            j_left = pred["j_left"]
            j_right = pred["j_right"]
            buttons = pred["buttons"]
            timings = pred["timings"]
            
            n_actions = len(buttons)
            
            env_actions = []
            
            for i in range(n_actions):
                move_action = zero_action.copy()
                
                xl, yl = j_left[i]
                xr, yr = j_right[i]
                move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int64)
                move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int64)
                move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int64)
                move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int64)
                
                button_vector = buttons[i]
                for name, value in zip(TOKEN_SET, button_vector):
                    if "TRIGGER" in name:
                        move_action[name] = np.array([int(value * 255)], dtype=np.int64)
                    else:
                        move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0
                
                env_actions.append(move_action)
            
            for i, a in enumerate(env_actions):
                if NO_MENU:
                    if a["START"]:
                        print("\n  [Blocked START button]")
                    a["GUIDE"] = 0
                    a["START"] = 0
                    a["BACK"] = 0
                
                obs, reward, terminated, truncated, info = env.step(action=a)
                
                if not args.no_record:
                    obs_viz = np.array(obs).copy()
                    clean_viz = cv2.resize(obs_viz, (1920, 1080), interpolation=cv2.INTER_AREA)
                    debug_viz = create_viz(
                        cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA),
                        i, j_left, j_right, buttons, token_set=TOKEN_SET
                    )
                    debug_recorder.add_frame(debug_viz)
                    clean_recorder.add_frame(clean_viz)
            
            with open(PATH_ACTIONS, "a") as f:
                for i, a in enumerate(env_actions):
                    # Convert controller payloads to JSON-safe values before appending logs.
                    action_log = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in a.items()
                    }
                    action_log["step"] = step_count
                    action_log["substep"] = i
                    json.dump(action_log, f)
                    f.write("\n")
            
            step_count += 1
            
            if step_count % 10 == 0 and timings:
                print(f"\n--- Performance ---")
                print(f"Vision: {timings['vision_encoder']:.2f}ms | "
                      f"DiT: {timings['dit_loop']:.2f}ms | "
                      f"Total: {timings['vision_encoder'] + timings['dit_loop']:.2f}ms")
            
            if step_count % 10 == 0:
                avg_infer = np.mean(inference_times[-10:]) * 1000
                print(f"Step {step_count:4d}: {n_actions:2d} actions, {avg_infer:5.1f}ms avg")
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        env.unpause()
        env.close()
        
        if debug_recorder:
            debug_recorder.close()
        if clean_recorder:
            clean_recorder.close()
        
        if inference_times:
            print("\n" + "="*80)
            print("STATISTICS")
            print("="*80)
            print(f"  Steps:     {step_count}")
            print(f"  Avg:       {np.mean(inference_times)*1000:.1f}ms")
            print(f"  Min/Max:   {np.min(inference_times)*1000:.1f} / {np.max(inference_times)*1000:.1f}ms")
            if len(inference_times) > 10:
                print(f"  P50/P95:   {np.percentile(inference_times, 50)*1000:.1f} / {np.percentile(inference_times, 95)*1000:.1f}ms")
            print("="*80)


if __name__ == "__main__":
    main()

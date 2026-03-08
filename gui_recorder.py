import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import json
import os
import pygame
import dxcam
import pywinctl as pwc
import ctypes
from datetime import datetime
from collections import deque
from queue import Queue
import keyboard

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

BASE_OUTPUT_DIR = "my_dataset_peppino"
FPS = 60
DEADZONE = 0.05
FRAME_INTERVAL = 1.0 / FPS
JPEG_QUALITY = 95

# ============================================
# RECORDER APP
# ============================================

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NitroGen Data Recorder v3 (Direct JPEG)")
        self.root.geometry("1000x900")

        self.is_recording = False
        self.is_counting_down = False
        self.camera = None
        self.joy = None
        self.target_window = None
        self.capture_region = None
        self.frame_count = 0
        self.stop_event = threading.Event()

        self.gamepad_buffer = deque(maxlen=120)
        self.gamepad_lock = threading.Lock()
        
        self.write_queue = None

        pygame.init()
        pygame.joystick.init()

        keyboard.add_hotkey('f9', self.toggle_recording)
        self.root.bind('<Escape>', lambda e: self.stop_recording() if self.is_recording else None)

        self._build_ui()
        self.refresh_all()
        self.combo_joy.bind("<<ComboboxSelected>>", self.on_joy_selected)
        self.combo_windows.bind("<<ComboboxSelected>>", self.on_window_selected)
        self.update_loop()

    # ============================================
    # UI BUILDING
    # ============================================

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill="x", pady=5)

        tk.Label(settings_frame, text="Game:").grid(row=0, column=0, sticky="w")
        self.combo_windows = ttk.Combobox(settings_frame, state="readonly", width=50)
        self.combo_windows.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(settings_frame, text="Gamepad:").grid(row=1, column=0, sticky="w")
        self.combo_joy = ttk.Combobox(settings_frame, state="readonly", width=50)
        self.combo_joy.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(settings_frame, text="Refresh", command=self.refresh_all).grid(
            row=0, column=2, rowspan=2, padx=10, sticky="nsew"
        )

        self.var_read_sticks = tk.BooleanVar(value=True)
        self.chk_sticks = tk.Checkbutton(
            settings_frame, text="Capture analog sticks (L/R)",
            variable=self.var_read_sticks, font=("Arial", 10, "bold")
        )
        self.chk_sticks.grid(row=2, column=1, sticky="w", pady=5)

        mid_frame = ttk.Frame(main_frame)
        mid_frame.pack(fill="both", expand=True, pady=10)

        self.preview_label = tk.Label(mid_frame, bg="black", text="PREVIEW", fg="white")
        self.preview_label.pack(side="left", padx=5, fill="both", expand=True)

        control_frame = ttk.LabelFrame(mid_frame, text="Input Monitor", padding="10")
        control_frame.pack(side="right", fill="y", padx=5)

        self.btn_indicators = {}
        btns = [
            ("A (South)", 0), ("B (East)", 1), ("X (West)", 2), ("Y (North)", 3),
            ("LB", 4), ("RB", 5), ("Back", 6), ("Start", 7), ("L3", 8), ("R3", 9)
        ]

        for name, idx in btns:
            f = tk.Frame(control_frame)
            f.pack(fill="x")
            tk.Label(f, text=name, width=12, anchor="w").pack(side="left")
            ind = tk.Label(f, bg="gray", width=3)
            ind.pack(side="right")
            self.btn_indicators[idx] = ind

        self.lbl_axis_l = tk.Label(control_frame, text="L: 0, 0", fg="blue", font=("Consolas", 10))
        self.lbl_axis_l.pack(pady=5)
        self.lbl_axis_r = tk.Label(control_frame, text="R: 0, 0", fg="green", font=("Consolas", 10))
        self.lbl_axis_r.pack(pady=5)
        self.lbl_trig = tk.Label(control_frame, text="T: L=0, R=0", fg="red", font=("Consolas", 10))
        self.lbl_trig.pack(pady=5)
        self.lbl_hat = tk.Label(control_frame, text="D-Pad: (0, 0)")
        self.lbl_hat.pack(pady=5)

        self.lbl_sync = tk.Label(control_frame, text="Sync: --", font=("Consolas", 9), fg="gray")
        self.lbl_sync.pack(pady=5)
        
        self.lbl_queue = tk.Label(control_frame, text="Queue: 0", font=("Consolas", 9), fg="gray")
        self.lbl_queue.pack(pady=5)

        self.btn_record = tk.Button(
            main_frame, text="Start Recording (F9)", bg="#cc0000", fg="white",
            font=("Arial", 16, "bold"), command=self.toggle_recording
        )
        self.btn_record.pack(fill="x", pady=10)

        self.lbl_timer = tk.Label(main_frame, text="00:00:00 | Frames: 0 | F9 - Start/Stop", font=("Consolas", 12))
        self.lbl_timer.pack()
        
        self.lbl_mode = tk.Label(main_frame, text="Mode: Direct JPEG", font=("Arial", 9), fg="green")
        self.lbl_mode.pack()

    # ============================================
    # WINDOW & GAMEPAD MANAGEMENT
    # ============================================

    def refresh_all(self):
        self.all_windows = [w for w in pwc.getAllWindows() if w.title.strip()]
        self.combo_windows['values'] = [w.title for w in self.all_windows]

        pygame.joystick.quit()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        self.combo_joy['values'] = [f"{i}: {j.get_name()}" for i, j in enumerate(self.joysticks)]

        if self.joysticks:
            self.combo_joy.current(0)
            self.on_joy_selected()

    def on_joy_selected(self, e=None):
        if self.joy:
            self.joy.quit()
        idx = self.combo_joy.current()
        if idx != -1:
            self.joy = self.joysticks[idx]
            self.joy.init()

    def on_window_selected(self, e=None):
        idx = self.combo_windows.current()
        if idx != -1:
            self.target_window = self.all_windows[idx]
            self.capture_region = (
                self.target_window.left,
                self.target_window.top,
                self.target_window.right,
                self.target_window.bottom
            )
            if self.camera:
                try:
                    self.camera.stop()
                except:
                    pass
            self.camera = dxcam.create(output_color="RGB")

    # ============================================
    # RECORDING CONTROL
    # ============================================

    def toggle_recording(self):
        if not self.is_recording and not self.is_counting_down:
            self.root.after(0, self.start_countdown)
        elif self.is_recording:
            self.root.after(0, self.stop_recording)

    def start_countdown(self):
        if not self.camera or not self.joy or not self.capture_region:
            messagebox.showerror("Error", "Select a game window and gamepad first.")
            return

        self.is_counting_down = True

        def countdown():
            for i in range(3, 0, -1):
                self.btn_record.config(text=f"Starting in {i}...", bg="orange")
                time.sleep(1)
            self.root.after(0, self.start_recording)

        threading.Thread(target=countdown, daemon=True).start()

    def start_recording(self):
        self.is_counting_down = False
        self.is_recording = True
        self.stop_event.clear()
        self.frame_count = 0
        self.dropped_frames = 0
        self.frames_written = 0

        # Keep an unbounded queue so capture naturally back-pressures on slow disk writes.
        self.write_queue = Queue()

        with self.gamepad_lock:
            self.gamepad_buffer.clear()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        self.frames_dir = os.path.join(self.current_run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.json_f = open(os.path.join(self.current_run_dir, "actions.jsonl"), "w")

        self.btn_record.config(text="Stop Recording (F9)", bg="black")

        threading.Thread(target=self.gamepad_poll_thread, daemon=True).start()
        threading.Thread(target=self.disk_writer_thread, daemon=True).start()
        threading.Thread(target=self.record_thread, daemon=True).start()

    def stop_recording(self):
        self.is_recording = False
        self.stop_event.set()
        self.btn_record.config(text="Saving...", state="disabled", bg="gray")

    # ============================================
    # GAMEPAD STATE CAPTURE
    # ============================================

    def apply_deadzone(self, val):
        return 0.0 if abs(val) < DEADZONE else val

    def capture_gamepad_state(self):
        if not self.joy:
            return None

        use_sticks = self.var_read_sticks.get()
        num_buttons = self.joy.get_numbuttons()

        if use_sticks:
            lx = self.apply_deadzone(self.joy.get_axis(0))
            ly = -self.apply_deadzone(self.joy.get_axis(1))
            rx = self.apply_deadzone(self.joy.get_axis(2))
            ry = -self.apply_deadzone(self.joy.get_axis(3))
        else:
            lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0

        lt_raw = (self.joy.get_axis(4) + 1) / 2
        rt_raw = (self.joy.get_axis(5) + 1) / 2
        lt_clean = 1.0 if lt_raw > 0.5 else 0.0
        rt_clean = 1.0 if rt_raw > 0.5 else 0.0

        hat = self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0)

        return {
            "buttons": {
                "SOUTH": self.joy.get_button(0),
                "EAST": self.joy.get_button(1),
                "WEST": self.joy.get_button(2),
                "NORTH": self.joy.get_button(3),
                "LEFT_SHOULDER": self.joy.get_button(4),
                "RIGHT_SHOULDER": self.joy.get_button(5),
                "BACK": self.joy.get_button(6),
                "START": self.joy.get_button(7),
                "LEFT_THUMB": self.joy.get_button(8) if num_buttons > 8 else 0,
                "RIGHT_THUMB": self.joy.get_button(9) if num_buttons > 9 else 0,
                "GUIDE": 0,
                "DPAD_UP": 1 if hat[1] == 1 else 0,
                "DPAD_DOWN": 1 if hat[1] == -1 else 0,
                "DPAD_LEFT": 1 if hat[0] == -1 else 0,
                "DPAD_RIGHT": 1 if hat[0] == 1 else 0
            },
            "sticks": {
                "AXIS_LEFTX": round(lx, 4),
                "AXIS_LEFTY": round(ly, 4),
                "AXIS_RIGHTX": round(rx, 4),
                "AXIS_RIGHTY": round(ry, 4),
                "LEFT_TRIGGER": lt_clean,
                "RIGHT_TRIGGER": rt_clean
            }
        }

    # ============================================
    # BACKGROUND THREADS
    # ============================================

    def gamepad_poll_thread(self):
        poll_interval = 0.002

        while not self.stop_event.is_set():
            pygame.event.pump()
            timestamp = time.perf_counter()
            state = self.capture_gamepad_state()

            if state:
                with self.gamepad_lock:
                    self.gamepad_buffer.append((timestamp, state))

            time.sleep(poll_interval)

    def get_gamepad_state_at_time(self, target_time):
        with self.gamepad_lock:
            if not self.gamepad_buffer:
                return None

            best_state = None
            best_diff = float('inf')

            for ts, state in self.gamepad_buffer:
                diff = abs(ts - target_time)
                if diff < best_diff:
                    best_diff = diff
                    best_state = state

            return best_state, best_diff

    def disk_writer_thread(self):
        """Write frames to disk. JSON stays on the timing-critical record thread."""
        write_times = []
        
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                item = self.write_queue.get(timeout=0.1)
            except:
                continue
            
            if item is None:
                break
                
            frame_idx, frame_bgr = item
            
            write_start = time.perf_counter()
            filename = os.path.join(self.frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            write_times.append(time.perf_counter() - write_start)
            
            self.frames_written += 1
        
        if write_times:
            self.write_stats = f"Write Avg: {np.mean(write_times)*1000:.2f}ms, Max: {np.max(write_times)*1000:.2f}ms"
        else:
            self.write_stats = "N/A"

    def record_thread(self):
        """Main recording thread - captures frames and synchronizes with gamepad state."""
        start_time = time.perf_counter()
        next_frame_time = start_time
        last_frame_img = None
        max_lag_frames = 3

        sync_diffs = []

        while not self.stop_event.is_set():
            current_time = time.perf_counter()

            while current_time < next_frame_time and not self.stop_event.is_set():
                sleep_time = next_frame_time - current_time
                if sleep_time > 0.002:
                    time.sleep(sleep_time - 0.001)
                current_time = time.perf_counter()

            if self.stop_event.is_set():
                break

            frame_capture_time = time.perf_counter()
            frame = self.camera.grab(region=self.capture_region)

            if frame is not None:
                last_frame_img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            elif last_frame_img is None:
                last_frame_img = np.zeros((256, 256, 3), dtype=np.uint8)

            gamepad_result = self.get_gamepad_state_at_time(frame_capture_time)

            if gamepad_result:
                gamepad_state, sync_diff = gamepad_result
                sync_diffs.append(sync_diff)
            else:
                pygame.event.pump()
                gamepad_state = self.capture_gamepad_state()
                sync_diff = 0

            if last_frame_img is not None and gamepad_state is not None:
                frame_bgr = cv2.cvtColor(last_frame_img, cv2.COLOR_RGB2BGR)
                
                # Queue the frame for disk IO. This blocks if the writer cannot keep up.
                self.write_queue.put((self.frame_count, frame_bgr.copy()))
                
                # Write JSON immediately to keep frame/action timing aligned.
                record = {
                    "frame": self.frame_count,
                    "timestamp": round(frame_capture_time - start_time, 6),
                    "sync_diff_ms": round(sync_diff * 1000, 2),
                    "actions": gamepad_state
                }
                self.json_f.write(json.dumps(record) + "\n")
                
                self.frame_count += 1

            next_frame_time += FRAME_INTERVAL

            lag = current_time - next_frame_time
            if lag > FRAME_INTERVAL * max_lag_frames:
                skipped = int(lag / FRAME_INTERVAL)
                next_frame_time += skipped * FRAME_INTERVAL
                self.dropped_frames += skipped

        # Finalize output files.
        self.write_queue.put(None)
        self.json_f.close()
        
        # Wait until the frame writer drains the queue.
        while self.frames_written < self.frame_count:
            time.sleep(0.1)

        if sync_diffs:
            avg_sync = np.mean(sync_diffs) * 1000
            max_sync = np.max(sync_diffs) * 1000
            self.sync_stats = f"Sync Avg: {avg_sync:.2f}ms, Max: {max_sync:.2f}ms"
        else:
            self.sync_stats = "N/A"

        self.root.after(0, self.on_finish)

    # ============================================
    # FINALIZATION & UI UPDATES
    # ============================================

    def on_finish(self):
        self.btn_record.config(text="Start Recording (F9)", bg="#cc0000", state="normal")

        total_size = 0
        if os.path.exists(self.frames_dir):
            for f in os.listdir(self.frames_dir):
                fpath = os.path.join(self.frames_dir, f)
                if os.path.isfile(fpath):
                    total_size += os.path.getsize(fpath)
        size_mb = total_size / (1024 * 1024)

        msg = f"📁 {self.current_run_dir}\n\n"
        msg += f"Frames: {self.frame_count}\n"
        msg += f"Dropped: {self.dropped_frames}\n"
        msg += f"Size: {size_mb:.1f} MB\n\n"
        msg += f"📊 {self.sync_stats}\n"
        msg += f"💿 {getattr(self, 'write_stats', 'N/A')}"

        messagebox.showinfo("Done", msg)

    def update_loop(self):
        if self.joy:
            pygame.event.pump()
            num_buttons = self.joy.get_numbuttons()
            use_sticks = self.var_read_sticks.get()

            for i, ind in self.btn_indicators.items():
                if i < num_buttons:
                    ind.config(bg="#00ff00" if self.joy.get_button(i) else "gray")

            if use_sticks:
                lx = self.joy.get_axis(0)
                ly = -self.joy.get_axis(1)
                rx = self.joy.get_axis(2)
                ry = -self.joy.get_axis(3)
            else:
                lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0

            if self.joy.get_numaxes() > 5:
                lt = (self.joy.get_axis(4) + 1) / 2
                rt = (self.joy.get_axis(5) + 1) / 2
            else:
                lt, rt = 0.0, 0.0

            self.lbl_axis_l.config(text=f"L: {lx:.2f}, {ly:.2f}")
            self.lbl_axis_r.config(text=f"R: {rx:.2f}, {ry:.2f}")
            self.lbl_trig.config(text=f"T: L={lt:.2f}, R={rt:.2f}")

            if self.joy.get_numhats() > 0:
                self.lbl_hat.config(text=f"D-Pad: {self.joy.get_hat(0)}")

        if self.is_recording and self.write_queue:
            q = self.write_queue.qsize()
            color = "green" if q < 30 else ("orange" if q < 60 else "red")
            self.lbl_queue.config(text=f"Queue: {q}", fg=color)

        if self.camera and self.capture_region and not self.is_recording:
            frame = self.camera.grab(region=self.capture_region)
            if frame is not None:
                img = Image.fromarray(cv2.resize(frame, (600, 340)))
                tkimg = ImageTk.PhotoImage(img)
                self.preview_label.config(image=tkimg)
                self.preview_label.image = tkimg

        if self.is_recording:
            total_sec = self.frame_count // FPS
            m, s = divmod(total_sec, 60)
            h, m = divmod(m, 60)
            self.lbl_timer.config(text=f"{h:02d}:{m:02d}:{s:02d} | Frames: {self.frame_count}")

        self.root.after(33, self.update_loop)


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()
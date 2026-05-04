"""
data_collector.py
=================
Tool to record a real-world webcam dataset for the benchmark suite.
Also provides a utility function `load_dataset()` to read the recorded MP4
into memory for the other benchmark scripts.
"""
from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np

def collect_dataset(output_path: Path, duration_sec: int = 15, fps: int = 30) -> None:
    """
    Open the webcam and record a video for `duration_sec` seconds.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam (VideoCapture(0)).")

    # Set common resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Read one frame to get actual resolution
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read from webcam.")

    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    total_frames = duration_sec * fps

    print(f"\n[RECORDING] Benchmark dataset to {output_path.name}")
    print(f"   Resolution: {w}x{h} @ {fps} FPS")
    print(f"   Duration:   {duration_sec} seconds")
    print("\n   Get ready! Recording starts in 3...")
    time.sleep(1)
    print("   2...")
    time.sleep(1)
    print("   1...")
    time.sleep(1)
    print("\n>>> RECORDING (Follow the on-screen instructions)...")

    frame_count = 0
    start_time = time.time()

    # Visual guide phases
    phases = [
        (0.0, 3.0, "1. Hold Hand in Center", (w//2, h//2)),
        (3.0, 6.0, "2. Slash to Top Right!", (w - 200, 200)),
        (6.0, 9.0, "3. Slash to Bottom Left!", (200, h - 200)),
        (9.0, 12.0, "4. Move Fast (Left <-> Right)", None),
        (12.0, 15.0, "5. Hide Hand (Move out of frame)", None),
    ]

    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame so it acts like a mirror when recording
            frame = cv2.flip(frame, 1)

            # --- Write clean frame to file (no overlays) ---
            out.write(frame)

            # --- Draw UI on the preview frame ---
            preview = frame.copy()
            elapsed = time.time() - start_time
            
            # Determine current phase
            current_instruction = ""
            target_pos = None
            for p_start, p_end, text, pos in phases:
                if p_start <= elapsed < p_end:
                    current_instruction = text
                    target_pos = pos
                    break
            
            # Draw Instruction text
            cv2.putText(preview, current_instruction, (50, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
            
            # Draw Target if any
            if target_pos:
                # Pulsing radius
                r = int(60 + 10 * np.sin(elapsed * 10))
                cv2.circle(preview, target_pos, r, (0, 255, 0), 4)
                cv2.circle(preview, target_pos, 5, (0, 0, 255), -1)

            # Draw progress bar
            progress = min(1.0, elapsed / duration_sec)
            bar_width = int(progress * w)
            cv2.rectangle(preview, (0, h - 20), (bar_width, h), (0, 0, 255), -1)
            cv2.putText(preview, f"Recording: {max(0, duration_sec - int(elapsed))}s left", 
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Food Ninja - Data Collector", preview)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("   Recording aborted early by user.")
                break
                
            frame_count += 1
            
            # Simple sync to target fps
            elapsed = time.time() - start_time
            expected_time = frame_count / fps
            if expected_time > elapsed:
                time.sleep(expected_time - elapsed)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Dataset saved -> {output_path}")


def load_dataset(video_path: Path | str) -> list[np.ndarray]:
    """
    Load all frames from an MP4 video into a list of NumPy arrays.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {video_path}\n"
            "   Please run `python -m foodninja.benchmark --collect` first to record a dataset."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

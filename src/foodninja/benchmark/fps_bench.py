"""
fps_bench.py
============
FPS benchmark for the Food Ninja tracking pipeline.

Measures three execution paths on synthetic (randomly generated) BGR frames:

  • ``camshift_only``  — CamShift measure (no Kalman, no MediaPipe)
  • ``kalman_camshift`` — Kalman predict + CamShift + Gating + Kalman update
  • ``full_pipeline``  — Complete HandTracker.process_frame() (excl. MediaPipe
                          background thread, which runs asynchronously and is
                          excluded from latency measurements)

Notes
-----
* AMD Radeon integrated graphics does not expose an NVIDIA CUDA device, so
  cv2.cuda / GPU-accelerated OpenCV is not available on this system.
  The report will note this and focus on the CPU-optimised paths.
* 30 warm-up frames are discarded before recording; 300 frames are timed.
"""
from __future__ import annotations

import json
import time
import statistics
from pathlib import Path

import time
import numpy as np

from foodninja.benchmark.data_collector import load_dataset
from foodninja.config import TrackingConfig
from foodninja.core.models import TrackingState, TrackMode
from foodninja.tracking.kalman_filter import KalmanFilterModule
from foodninja.tracking.camshift import CamShiftModule
from foodninja.tracking.gating import accept_measurement
from foodninja.tracking.tracker import HandTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 1280, 720
WARMUP = 30
MEASURE = 300
DT = 1.0 / 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_camshift_module(frame: np.ndarray) -> tuple[CamShiftModule, TrackingState]:
    """Initialise a CamShiftModule with a skin ROI."""
    cs = CamShiftModule()
    import cv2
    h, w = frame.shape[:2]
    # Assume hand is near center for initial tracking start
    roi = (w // 2 - 60, h // 2 - 80, 120, 160)
    cs.set_target(frame, roi)
    state = TrackingState(cx=w / 2, cy=h / 2, vx=0, vy=0, width=120, height=160)
    return cs, state


def _benchmark(fn, frames: list[np.ndarray], warmup: int) -> dict:
    """Run *fn(frame)* for each frame, discard warm-up, return stats."""
    for i in range(warmup):
        fn(frames[i % len(frames)])

    times_ns: list[int] = []
    for frame in frames:
        t0 = time.perf_counter_ns()
        fn(frame)
        times_ns.append(time.perf_counter_ns() - t0)

    times_ms = [t / 1_000_000 for t in times_ns]
    fps_vals = [1000 / t for t in times_ms if t > 0]

    return {
        "n_frames": len(times_ms),
        "mean_fps": round(statistics.mean(fps_vals), 2),
        "p50_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(times_ms, 99)), 3),
        "std_ms": round(statistics.stdev(times_ms), 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
    }


# ---------------------------------------------------------------------------
# Benchmark paths
# ---------------------------------------------------------------------------

def bench_camshift_only(frames: list[np.ndarray]) -> dict:
    """CamShift measure only (no Kalman, no MediaPipe)."""
    cs, state = _init_camshift_module(frames[0])
    trace_p = 100.0

    def _run(frame):
        cs.measure(frame, state, trace_p, velocity_lookahead=0.5, dt=DT)

    stats = _benchmark(_run, frames, WARMUP)
    stats["path"] = "camshift_only"
    stats["description"] = "CamShift colour-histogram measure (CPU)"
    return stats


def bench_kalman_camshift(frames: list[np.ndarray]) -> dict:
    """Kalman predict + CamShift + Gating + Kalman update (no MediaPipe)."""
    cs, _ = _init_camshift_module(frames[0])
    config = TrackingConfig()
    kf = KalmanFilterModule()
    h, w = frames[0].shape[:2]
    initial = np.array([w / 2, h / 2, 0, 0, 120, 160], dtype=np.float32)
    kf.initialize(initial)

    def _run(frame):
        x_pred, P_pred = kf.predict()
        pred_state = TrackingState.from_array(x_pred)
        trace_p = float(np.trace(P_pred[:2, :2]))
        z_t = cs.measure(frame, pred_state, trace_p, velocity_lookahead=0.5, dt=DT)
        if z_t is not None:
            accepted = accept_measurement(
                predicted_state=pred_state,
                measurement=z_t,
                position_variance_x=P_pred[0, 0],
                position_variance_y=P_pred[1, 1],
                config=config,
            )
            if accepted:
                kf.update(z_t.to_array())

    stats = _benchmark(_run, frames, WARMUP)
    stats["path"] = "kalman_camshift"
    stats["description"] = "Kalman predict + CamShift + Gating + Kalman update (CPU)"
    return stats


def bench_full_pipeline(frames: list[np.ndarray], model_path: str | None = None) -> dict:
    """
    Full HandTracker.process_frame() — MediaPipe runs asynchronously in a
    background thread and its timing is excluded from per-frame measurements.
    """
    config = TrackingConfig()
    if model_path:
        config.mediapipe_model_path = model_path

    try:
        tracker = HandTracker(config, profiling_enabled=False)
        # WARMUP: Force MediaPipe to initialize the tracker state
        for frame in frames[:30]:
            tracker.process_frame(frame, 0.033)
            time.sleep(0.05)
            if tracker.mode == TrackMode.TRACKING:
                break
    except Exception as exc:
        return {
            "path": "full_pipeline",
            "description": "Full HandTracker pipeline (CPU)",
            "error": str(exc),
        }

    def _run(frame):
        tracker.process_frame(frame, dt=DT)

    stats = _benchmark(_run, frames, WARMUP)
    stats["path"] = "full_pipeline"
    stats["description"] = "Full HandTracker pipeline (CPU, excl. async MediaPipe)"
    return stats


# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------

def check_gpu() -> dict:
    """Report GPU backend availability (CUDA / OpenCL)."""
    info: dict = {"cuda_available": False, "opencl_available": False, "note": ""}

    # CUDA (NVIDIA only — not available on AMD Radeon integrated)
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
        info["cuda_available"] = n > 0
        if n > 0:
            info["cuda_device_count"] = n
    except Exception:
        pass

    # OpenCL (may be available on AMD via ROCm or Windows OpenCL)
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            info["opencl_available"] = cv2.ocl.useOpenCL()
    except Exception:
        pass

    if not info["cuda_available"]:
        info["note"] = (
            "CUDA not available (AMD Radeon integrated GPU). "
            "All benchmarks run on CPU. "
            "For NVIDIA GPU benchmarks, install opencv-contrib-python with CUDA build."
        )

    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(model_path: str | None = None, dataset_path: Path | None = None) -> dict:
    """Run all FPS benchmarks and return combined results."""
    gpu_info = check_gpu()

    try:
        frames = load_dataset(dataset_path)
    except Exception as exc:
        return {
            "gpu": gpu_info,
            "results": [{"path": "dataset", "error": str(exc)}]
        }
    
    print(f"  Loaded {len(frames)} frames from dataset.")
    print(f"  GPU: CUDA={'yes' if gpu_info['cuda_available'] else 'no'}  "
          f"OpenCL={'yes' if gpu_info['opencl_available'] else 'no'}")

    print("  [1/3] CamShift only …")
    r_cs = bench_camshift_only(frames)

    print(f"        → {r_cs['mean_fps']:.1f} FPS  (p95 {r_cs['p95_ms']} ms)")

    print("  [2/3] Kalman + CamShift …")
    r_kc = bench_kalman_camshift(frames)
    print(f"        → {r_kc['mean_fps']:.1f} FPS  (p95 {r_kc['p95_ms']} ms)")

    print("  [3/3] Full pipeline …")
    r_fp = bench_full_pipeline(frames, model_path=model_path)
    if "error" not in r_fp:
        print(f"        → {r_fp['mean_fps']:.1f} FPS  (p95 {r_fp['p95_ms']} ms)")
    else:
        print(f"        ✗ Error: {r_fp['error']}")

    return {
        "gpu": gpu_info,
        "results": [r_cs, r_kc, r_fp],
    }

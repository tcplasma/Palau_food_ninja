"""
model_compare.py
================
Model size vs accuracy tradeoff for the three MediaPipe hand-landmarker
variants (Lite / Full / Heavy).

Metrics collected per variant
------------------------------
  • model_size_mb   — on-disk .task file size
  • inference_ms    — mean / p50 / p95 latency (background thread excluded;
                      measures synchronous HandDetector.detect_hand() only)
  • detection_rate  — fraction of synthetic frames where a hand is detected
  • bbox_stability  — std-dev of bounding-box centre-x across detected frames
                      (lower = more stable, less jitter)

Output
------
  benchmark_models.json  — raw stats
  model_tradeoff.png     — scatter plot: inference latency vs model size,
                           bubble area ∝ detection rate
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from foodninja.benchmark.data_collector import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480   # smaller crop — mirrors MediaPipe ROI logic
N_FRAMES = 80                  # frames per variant (kept low for CI speed)
WARMUP = 10

VARIANT_ORDER = ["lite", "full", "heavy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_variant(model_path: Path, variant: str, frames: list[np.ndarray]) -> dict:
    """Run synchronous HandDetector on *frames* and collect metrics."""
    from foodninja.initialization.mediapipe_recovery import HandDetector

    size_mb = round(model_path.stat().st_size / 1024 / 1024, 2)
    print(f"    [{variant:5s}] {model_path.name}  ({size_mb} MB)")

    try:
        detector = HandDetector(str(model_path))
    except Exception as exc:
        return {
            "variant": variant,
            "model_size_mb": size_mb,
            "error": str(exc),
        }

    inference_times_ms: list[float] = []
    detected_cx: list[float] = []
    n_detected = 0

    # Warm-up
    for frame in frames[:WARMUP]:
        detector.detect_hand(frame)

    # Measurement
    for frame in frames:
        t0 = time.perf_counter_ns()
        result = detector.detect_hand(frame)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        inference_times_ms.append(elapsed_ms)
        if result is not None:
            n_detected += 1
            detected_cx.append(result.cx)

    detector.close()

    detection_rate = round(n_detected / len(frames), 3) if frames else 0.0
    bbox_stability = round(float(np.std(detected_cx)), 2) if detected_cx else 0.0

    result_dict = {
        "variant": variant,
        "model_size_mb": size_mb,
        "detection_rate": detection_rate,
        "bbox_stability_sigma_px": bbox_stability,
        "n_frames": len(frames),
        "n_detected": n_detected,
        "inference_mean_ms": round(statistics.mean(inference_times_ms), 2),
        "inference_p50_ms": round(float(np.percentile(inference_times_ms, 50)), 2),
        "inference_p95_ms": round(float(np.percentile(inference_times_ms, 95)), 2),
        "inference_std_ms": round(statistics.stdev(inference_times_ms), 2) if len(inference_times_ms) > 1 else 0.0,
    }

    print(f"           inference {result_dict['inference_mean_ms']:.1f} ms  "
          f"detect {detection_rate*100:.0f}%  "
          f"σ_cx {bbox_stability:.1f} px")

    return result_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(models_dir: Path, available_variants: dict[str, Path] | None = None, dataset_path: Path | None = None) -> dict:
    """
    Compare lite / full / heavy model variants.

    Parameters
    ----------
    models_dir:
        Directory that contains the .task files.
    available_variants:
        Optional pre-resolved {variant: Path} mapping from model_downloader.
        If None, we scan models_dir for known filenames.
    """
    from foodninja.benchmark.model_downloader import MODELS

    try:
        frames = load_dataset(dataset_path)
    except Exception as exc:
        return {"error": str(exc)}

    # Resolve paths
    if available_variants is None:
        available_variants = {}
        for variant, info in MODELS.items():
            p = models_dir / info["filename"]
            if p.exists():
                available_variants[variant] = p

    if not available_variants:
        return {"error": "No model files found. Run with --download-models first."}

    # Use dataset frames instead of synthetic frames
    # Limit to N_FRAMES to keep tests fast, but we need WARMUP as well
    total_needed = WARMUP + N_FRAMES
    if len(frames) > total_needed:
        # Take a subset if the video is very long
        frames = frames[:total_needed]

    bench_frames = frames[WARMUP:] if len(frames) > WARMUP else frames

    results = []
    for variant in VARIANT_ORDER:
        if variant not in available_variants:
            print(f"    [{variant:5s}] not available — skipping")
            continue
        r = _measure_variant(available_variants[variant], variant, bench_frames)
        results.append(r)

    return {"variants": results}


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------

def render_chart(results: dict, output_path: Path) -> None:
    """Scatter plot: model size (x) vs inference latency (y), bubble = detect rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed — skipping chart")
        return

    variants = [v for v in results.get("variants", []) if "error" not in v]
    if not variants:
        return

    sizes = [v["model_size_mb"] for v in variants]
    latencies = [v["inference_mean_ms"] for v in variants]
    rates = [v["detection_rate"] for v in variants]
    labels = [v["variant"].capitalize() for v in variants]

    bubble_areas = [max(r, 0.05) * 3500 for r in rates]
    colours = ["#4cc9f0", "#f77f00", "#e63946"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (x, y, a, label, colour) in enumerate(zip(sizes, latencies, bubble_areas, labels, colours)):
        sc = ax.scatter(x, y, s=a, color=colour, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.7)
        ax.annotate(
            f"{label}\n{rates[i]*100:.0f}% detect",
            (x, y), textcoords="offset points", xytext=(10, 6),
            color="white", fontsize=9,
        )

    ax.set_xlabel("Model Size (MB)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Inference Latency (ms)", color="#aaaaaa", fontsize=10)
    ax.set_title("Model Size vs Accuracy Tradeoff\n(bubble area = detection rate)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333355")
    ax.grid(color="#333355", linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart saved → {output_path}")

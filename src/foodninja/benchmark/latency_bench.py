"""
latency_bench.py
================
Per-stage latency analysis of the Food Ninja tracking pipeline.

Instruments every stage inside HandTracker.process_frame() using the
ProfilingContext added to TrackingStepResult, then produces:

  • ``benchmark_latency.json``   — raw stats
  • ``latency_breakdown.png``    — horizontal waterfall bar chart

Stages measured
---------------
  1. kalman_predict    — F·x, F·P·F^T + Q
  2. camshift_measure  — HSV back-project → CamShift → multi-metric confidence
  3. gating            — Mahalanobis distance gate
  4. kalman_update     — Kalman gain + state correction
  5. one_euro          — 1€ filter smooth for visual output
  6. total_tracking    — sum of above (true end-to-end pipeline ns)
"""
from __future__ import annotations

import json
import statistics
import time
import numpy as np
from pathlib import Path

from foodninja.benchmark.data_collector import load_dataset
from foodninja.config import TrackingConfig
from foodninja.core.models import TrackingState, TrackMode
from foodninja.tracking.tracker import HandTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 1280, 720
WARMUP = 30
MEASURE = 200
DT = 1.0 / 60.0

STAGE_KEYS = [
    ("t_kalman_predict_ns", "Kalman Predict"),
    ("t_camshift_measure_ns", "CamShift Measure"),
    ("t_gating_ns", "Gating"),
    ("t_kalman_update_ns", "Kalman Update"),
    ("t_one_euro_ns", "1€ Filter"),
    ("t_total_tracking_ns", "Total Tracking"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns_to_us(ns: int) -> float:
    return ns / 1_000.0


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "mean_us": round(statistics.mean(values), 2),
        "p50_us": round(float(np.percentile(values, 50)), 2),
        "p95_us": round(float(np.percentile(values, 95)), 2),
        "p99_us": round(float(np.percentile(values, 99)), 2),
        "std_us": round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
        "min_us": round(min(values), 2),
        "max_us": round(max(values), 2),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(model_path: str | None = None, dataset_path: Path | None = None) -> dict:
    """Run per-stage latency benchmark and return results dict."""
    try:
        frames = load_dataset(dataset_path)
    except Exception as exc:
        return {"error": str(exc)}
    
    print("  Initialising HandTracker with profiling=True …")
    config = TrackingConfig()
    if model_path:
        config.mediapipe_model_path = model_path

    try:
        tracker = HandTracker(config, profiling_enabled=True)
        # WARMUP: Force MediaPipe to initialize the tracker state
        for frame in frames[:30]:
            tracker.process_frame(frame, 0.033)
            time.sleep(0.05)
            if tracker.mode == TrackMode.TRACKING:
                break
    except Exception as exc:
        return {"error": str(exc)}

    # Accumulate per-stage ns lists
    stage_samples: dict[str, list[float]] = {key: [] for key, _ in STAGE_KEYS}

    # Warm-up
    for frame in frames[:WARMUP]:
        tracker.process_frame(frame, dt=DT)

    # Measurement
    for frame in frames[WARMUP:]:
        result = tracker.process_frame(frame, dt=DT)
        if result.profiling is not None:
            p = result.profiling
            for attr, _ in STAGE_KEYS:
                ns_val = getattr(p, attr, 0)
                if ns_val > 0:
                    stage_samples[attr].append(_ns_to_us(ns_val))

    # Build stats report
    stages_out = []
    for attr, label in STAGE_KEYS:
        s = _stats(stage_samples[attr])
        s["stage"] = label
        s["n_samples"] = len(stage_samples[attr])
        stages_out.append(s)
        if s.get("mean_us"):
            print(f"    {label:22s}  {s['mean_us']:7.1f} us  (p95 {s['p95_us']} us)")

    return {"stages": stages_out, "n_frames": len(frames)}


# ---------------------------------------------------------------------------
# Chart renderer (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def render_chart(results: dict, output_path: Path) -> None:
    """Render a horizontal bar chart of mean per-stage latency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("    matplotlib not installed — skipping chart")
        return

    stages = [s for s in results.get("stages", []) if s.get("stage") != "Total Tracking"]
    if not stages:
        return

    labels = [s["stage"] for s in stages]
    means = [s.get("mean_us", 0) for s in stages]
    p95s = [s.get("p95_us", 0) for s in stages]

    # Colour palette
    colours = ["#4c9be8", "#e87c4c", "#62c87a", "#b07ae8", "#e8c84c"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, means, color=[colours[i % len(colours)] for i in range(len(labels))],
                   height=0.55, zorder=3)
    ax.barh(y_pos, p95s, color="none", edgecolor="white", linewidth=0.8,
            height=0.55, linestyle="--", zorder=4, alpha=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color="white", fontsize=10)
    ax.set_xlabel("Latency (µs)", color="#aaaaaa", fontsize=10)
    ax.set_title("Per-Stage Latency Breakdown", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333355")
    ax.grid(axis="x", color="#333355", linestyle="--", linewidth=0.6, zorder=0)

    for bar, val in zip(bars, means):
        ax.text(val + max(means) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} µs", va="center", color="white", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=colours[i % len(colours)], label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.2,
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart saved → {output_path}")

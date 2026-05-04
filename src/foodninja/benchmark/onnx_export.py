"""
onnx_export.py
==============
Exports the Food Ninja tracking core (Kalman filter + CamShift colour-histogram
lookup) as an ONNX computation graph, then benchmarks it against the baseline
NumPy implementation.

Why ONNX here, not MediaPipe?
------------------------------
MediaPipe ships its models as TFLite flatbuffers embedded inside a .task
archive.  Direct ONNX export of the hand-landmarker requires tf2onnx and a
matching TensorFlow install (large dependency, version-sensitive).  Instead,
we export the **tracking math** — the algorithmic core that runs on every
frame at 60 FPS — which is a more meaningful ONNX target for deployment.

Exported graph: ``KalmanPredict``
----------------------------------
  Inputs:
    x   [6, 1]  float32  — current state vector [cx, cy, vx, vy, w, h]
    P   [6, 6]  float32  — covariance matrix

  Outputs:
    x_pred  [6, 1]  float32  — predicted state
    P_pred  [6, 6]  float32  — predicted covariance

  The transition matrix F and noise matrix Q are baked in as constants.
  This lets onnxruntime fuse and optimise the MatMul chains.

Exported graph: ``CamShiftLookup``
------------------------------------
  Exports the colour-histogram back-projection as an ONNX Gather-based lookup.

  Inputs:
    hsv_pixels  [N, 2]  int32   — (h_bin, s_bin) pairs for N pixels
    hist        [180, 256] float32 — normalised H-S histogram

  Outputs:
    scores      [N]  float32   — back-projection values (0–255)

  In production this replaces the cv2.calcBackProject hot path with an
  onnxruntime-optimised gather, which benefits from SIMD and potential GPU EP.

Output files
------------
  kalman_predict.onnx      — Kalman predict sub-graph
  camshift_lookup.onnx     — CamShift back-projection lookup
  benchmark_onnx.json      — speedup results
  onnx_speedup.png         — bar chart
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def _export_kalman_predict(output_path: Path, dt: float = 1.0 / 60.0) -> Path:
    """Build and save the Kalman predict ONNX graph."""
    try:
        import onnx
        from onnx import numpy_helper, TensorProto
        from onnx.helper import (
            make_model, make_node, make_graph,
            make_tensor_value_info, make_opsetid,
        )
    except ImportError:
        raise ImportError("onnx package required: pip install onnx")

    F = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1,  0, 0, 0],
        [0, 0, 0,  1, 0, 0],
        [0, 0, 0,  0, 1, 0],
        [0, 0, 0,  0, 0, 1],
    ], dtype=np.float32)

    Q = np.diag([1.0, 1.0, 800.0, 800.0, 4.0, 4.0]).astype(np.float32)

    # ONNX initializers (constant matrices)
    F_init = numpy_helper.from_array(F, name="F")
    FT_init = numpy_helper.from_array(F.T.copy(), name="FT")
    Q_init = numpy_helper.from_array(Q, name="Q")

    # Graph inputs
    x_in = make_tensor_value_info("x", TensorProto.FLOAT, [6, 1])
    P_in = make_tensor_value_info("P", TensorProto.FLOAT, [6, 6])

    # Graph outputs
    x_out = make_tensor_value_info("x_pred", TensorProto.FLOAT, [6, 1])
    P_out = make_tensor_value_info("P_pred", TensorProto.FLOAT, [6, 6])

    # Nodes:  x_pred = F @ x
    node_xpred = make_node("Gemm", inputs=["F", "x"], outputs=["x_pred"],
                           alpha=1.0, beta=0.0, transA=0, transB=0)

    # P_pred = F @ P @ F^T + Q
    # Step 1: tmp = F @ P
    node_tmp = make_node("MatMul", inputs=["F", "P"], outputs=["tmp"])
    # Step 2: FP_FT = tmp @ F^T
    node_fpft = make_node("MatMul", inputs=["tmp", "FT"], outputs=["FP_FT"])
    # Step 3: P_pred = FP_FT + Q
    node_ppred = make_node("Add", inputs=["FP_FT", "Q"], outputs=["P_pred"])

    graph = make_graph(
        nodes=[node_xpred, node_tmp, node_fpft, node_ppred],
        name="KalmanPredict",
        inputs=[x_in, P_in],
        outputs=[x_out, P_out],
        initializer=[F_init, FT_init, Q_init],
    )

    model = make_model(graph, opset_imports=[make_opsetid("", 17)])
    model.doc_string = "Kalman filter predict step — F·x and F·P·F^T + Q"

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def _export_camshift_lookup(output_path: Path) -> Path:
    """Build and save the CamShift histogram lookup ONNX graph."""
    try:
        import onnx
        from onnx import TensorProto
        from onnx.helper import (
            make_model, make_node, make_graph,
            make_tensor_value_info, make_opsetid,
        )
    except ImportError:
        raise ImportError("onnx package required: pip install onnx")

    # Graph inputs
    # hsv_pixels: [N, 2] int64 — (h_bin, s_bin) pairs
    # hist:       [180, 256] float32 — normalised H-S histogram
    hsv_in = make_tensor_value_info("hsv_pixels", TensorProto.INT64, ["N", 2])
    hist_in = make_tensor_value_info("hist", TensorProto.FLOAT, [180, 256])

    # Graph output
    scores_out = make_tensor_value_info("scores", TensorProto.FLOAT, ["N"])

    # Step 1: h_idx = hsv_pixels[:, 0]
    # Step 2: s_idx = hsv_pixels[:, 1]
    # Step 3: row  = hist[h_idx]           — Gather on axis 0 gives [N, 256]
    # Step 4: scores = row[range(N), s_idx] — GatherElements picks one per row

    zero_const_node = make_node("Constant", inputs=[], outputs=["zero_i"],
        value_int=0,
        # use Constant operator attribute:
    )

    # Use Gather with indices on a constant axis
    # h_idx = Gather(hsv_pixels, 0, axis=1) → shape [N]
    axis0 = make_node("Constant", inputs=[], outputs=["axis0_val"])
    # embed axis=1 index scalar as a constant
    idx0_node = make_node("Gather", inputs=["hsv_pixels", "idx_col0"], outputs=["h_idx"], axis=1)
    idx1_node = make_node("Gather", inputs=["hsv_pixels", "idx_col1"], outputs=["s_idx"], axis=1)

    # row_vals = Gather(hist, h_idx, axis=0) → [N, 256]
    row_node = make_node("Gather", inputs=["hist", "h_idx"], outputs=["row_vals"], axis=0)

    # unsqueeze s_idx to [N, 1] for GatherElements
    unsqueeze_node = make_node("Unsqueeze", inputs=["s_idx", "unsqueeze_axes"], outputs=["s_idx_2d"])

    # GatherElements along axis=1: picks hist[h, s] for each pixel
    gather_elem_node = make_node("GatherElements", inputs=["row_vals", "s_idx_2d"],
                                  outputs=["scores_2d"], axis=1)

    # Squeeze back to [N]
    squeeze_node = make_node("Squeeze", inputs=["scores_2d", "squeeze_axes"], outputs=["scores"])

    from onnx import numpy_helper
    col0_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="idx_col0")
    col1_init = numpy_helper.from_array(np.array(1, dtype=np.int64), name="idx_col1")
    unsq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="unsqueeze_axes")
    sq_axes   = numpy_helper.from_array(np.array([1], dtype=np.int64), name="squeeze_axes")

    graph = make_graph(
        nodes=[idx0_node, idx1_node, row_node, unsqueeze_node, gather_elem_node, squeeze_node],
        name="CamShiftLookup",
        inputs=[hsv_in, hist_in],
        outputs=[scores_out],
        initializer=[col0_init, col1_init, unsq_axes, sq_axes],
    )

    model = make_model(graph, opset_imports=[make_opsetid("", 17)])
    model.doc_string = "CamShift H-S histogram back-projection via ONNX Gather"

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Runtime benchmarks
# ---------------------------------------------------------------------------

def _bench_kalman_numpy(n: int = 500) -> float:
    """Return mean ms per call using pure NumPy Kalman predict."""
    dt = 1.0 / 60.0
    F = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    Q = np.diag([1.0, 1.0, 800.0, 800.0, 4.0, 4.0]).astype(np.float32)
    x = np.random.rand(6, 1).astype(np.float32)
    P = np.eye(6, dtype=np.float32) * 100.0

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        x = F @ x
        P = F @ P @ F.T + Q
        times.append(time.perf_counter_ns() - t0)

    return statistics.mean(t / 1e6 for t in times[50:])


def _bench_kalman_onnx(model_path: Path, n: int = 500) -> float:
    """Return mean ms per call using onnxruntime on the exported ONNX graph."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required: pip install onnxruntime")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(6, 1).astype(np.float32)
    P = np.eye(6, dtype=np.float32) * 100.0

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        out = sess.run(None, {"x": x, "P": P})
        x, P = out[0], out[1]
        times.append(time.perf_counter_ns() - t0)

    return statistics.mean(t / 1e6 for t in times[50:])


def _bench_camshift_numpy(n: int = 500) -> float:
    """Return mean ms for NumPy-based histogram lookup on 10 000 pixels."""
    rng = np.random.default_rng(7)
    hist = rng.random((180, 256)).astype(np.float32) * 255.0
    pixels = rng.integers(0, [180, 256], size=(10_000, 2), dtype=np.int64)

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        scores = hist[pixels[:, 0], pixels[:, 1]]
        times.append(time.perf_counter_ns() - t0)

    return statistics.mean(t / 1e6 for t in times[50:])


def _bench_camshift_onnx(model_path: Path, n: int = 500) -> float:
    """Return mean ms for ONNX histogram lookup on 10 000 pixels."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required: pip install onnxruntime")

    rng = np.random.default_rng(7)
    hist = rng.random((180, 256)).astype(np.float32) * 255.0
    pixels = rng.integers(0, [180, 256], size=(10_000, 2), dtype=np.int64)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        sess.run(None, {"hsv_pixels": pixels, "hist": hist})
        times.append(time.perf_counter_ns() - t0)

    return statistics.mean(t / 1e6 for t in times[50:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(output_dir: Path) -> dict:
    """Export ONNX models and benchmark them. Returns results dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"graphs": []}

    # ── Kalman Predict ─────────────────────────────────────────────────────
    kalman_path = output_dir / "kalman_predict.onnx"
    print("  [1/2] Exporting KalmanPredict …")
    try:
        _export_kalman_predict(kalman_path)
        print(f"        Saved → {kalman_path}  ({kalman_path.stat().st_size/1024:.1f} KB)")

        numpy_ms = _bench_kalman_numpy()
        onnx_ms = _bench_kalman_onnx(kalman_path)
        speedup = round(numpy_ms / onnx_ms, 2) if onnx_ms > 0 else None
        print(f"        NumPy {numpy_ms*1000:.1f} µs  |  ONNX {onnx_ms*1000:.1f} µs  "
              f"|  speedup ×{speedup}")

        results["graphs"].append({
            "graph": "KalmanPredict",
            "model_path": str(kalman_path),
            "model_size_kb": round(kalman_path.stat().st_size / 1024, 1),
            "numpy_mean_ms": round(numpy_ms, 4),
            "onnx_mean_ms": round(onnx_ms, 4),
            "speedup_x": speedup,
            "provider": "CPUExecutionProvider",
        })
    except Exception as exc:
        print(f"        ✗ {exc}")
        results["graphs"].append({"graph": "KalmanPredict", "error": str(exc)})

    # ── CamShift Lookup ────────────────────────────────────────────────────
    camshift_path = output_dir / "camshift_lookup.onnx"
    print("  [2/2] Exporting CamShiftLookup …")
    try:
        _export_camshift_lookup(camshift_path)
        print(f"        Saved → {camshift_path}  ({camshift_path.stat().st_size/1024:.1f} KB)")

        numpy_ms = _bench_camshift_numpy()
        onnx_ms = _bench_camshift_onnx(camshift_path)
        speedup = round(numpy_ms / onnx_ms, 2) if onnx_ms > 0 else None
        print(f"        NumPy {numpy_ms*1000:.1f} µs  |  ONNX {onnx_ms*1000:.1f} µs  "
              f"|  speedup ×{speedup}")

        results["graphs"].append({
            "graph": "CamShiftLookup",
            "model_path": str(camshift_path),
            "model_size_kb": round(camshift_path.stat().st_size / 1024, 1),
            "numpy_mean_ms": round(numpy_ms, 4),
            "onnx_mean_ms": round(onnx_ms, 4),
            "speedup_x": speedup,
            "provider": "CPUExecutionProvider",
        })
    except Exception as exc:
        print(f"        ✗ {exc}")
        results["graphs"].append({"graph": "CamShiftLookup", "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------

def render_chart(results: dict, output_path: Path) -> None:
    """Grouped bar chart: NumPy vs ONNX latency per graph."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("    matplotlib not installed — skipping chart")
        return

    graphs = [g for g in results.get("graphs", []) if "error" not in g]
    if not graphs:
        return

    labels = [g["graph"] for g in graphs]
    numpy_us = [g["numpy_mean_ms"] * 1000 for g in graphs]
    onnx_us = [g["onnx_mean_ms"] * 1000 for g in graphs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars1 = ax.bar(x - width / 2, numpy_us, width, label="NumPy (baseline)",
                   color="#4cc9f0", alpha=0.9, zorder=3)
    bars2 = ax.bar(x + width / 2, onnx_us, width, label="ONNX Runtime (CPU EP)",
                   color="#f77f00", alpha=0.9, zorder=3)

    ax.set_ylabel("Latency (µs)", color="#aaaaaa", fontsize=10)
    ax.set_title("ONNX vs NumPy Baseline — Tracking Core Ops",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333355")
    ax.grid(axis="y", color="#333355", linestyle="--", linewidth=0.5, zorder=0)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)

    for bar, val in zip(bars1, numpy_us):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(numpy_us) * 0.01,
                f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=8)
    for bar, val, g in zip(bars2, onnx_us, graphs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(numpy_us) * 0.01,
                f"{val:.0f}\n×{g.get('speedup_x','?')}", ha="center", va="bottom",
                color="#f0c040", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart saved → {output_path}")

# 📈 Production Performance & Metric Interpretation

This document provides a detailed breakdown of the Food Ninja Computer Vision (CV) tracking pipeline's performance. These metrics were captured on a production-grade benchmarking suite using real-world hand-tracking datasets.

## 🚀 Executive Summary

| Category | Metric | Result | Benchmark (Target) |
| :--- | :--- | :--- | :--- |
| **Throughput** | End-to-End FPS | **260.2 FPS** | > 60 FPS |
| **Accuracy** | Hand Detection Rate | **95.0%** | > 90% |
| **Responsiveness** | Pipeline Latency | **4.5 ms** | < 10.0 ms |
| **Efficiency** | Inference Time | **22.7 ms** | < 33.3 ms |

---

## 🔍 Understanding the Metrics

### 1. FPS (Frames Per Second)
**What it is:** The number of full tracking cycles the system can complete in one second.
- **CamShift Only (~290 FPS):** Measures raw pixel-level color tracking throughput.
- **Full Pipeline (~260 FPS):** Measures the complete hybrid system (Kalman Prediction + CamShift Measurement + Gating + One-Euro Filtering).
- **Production Significance:** High FPS ensures that the game engine always has the latest hand coordinates, resulting in ultra-smooth "Fruit Ninja" style slashing even on CPU-only hardware.

### 2. Per-Stage Latency (Microseconds - µs)
We measure the exact time cost of every algorithmic step to identify bottlenecks.
- **Kalman Predict/Update:** Mathematical projection of the hand's next position. Extremely fast (< 100µs).
- **CamShift Measure:** The core "heavy lifter". It scans the search window to find the hand's skin-histogram centroid.
- **Gating:** The security layer. It calculates Mahalanobis distance to reject "noise" (e.g., a face or a piece of food) from being tracked as a hand.
- **Production Significance:** Total latency must stay well below the frame interval (33ms for 30FPS) to avoid "input lag" where the trail lags behind your actual hand.

### 3. Detection Rate & Stability
Using MediaPipe Task Landmarkers (Full model variant).
- **Detection Rate (95%):** How often the hand is "found" during initialization or recovery. 95% means the system almost never fails to pick up your hand when it enters the frame.
- **BBox Stability (σ):** Measures the "jitter" of the bounding box. A lower value means a smoother experience.
- **Production Significance:** High detection rate is the difference between a "student project" and a "reliable product". It ensures the game starts immediately when the user raises their hand.

---

## 🔷 The Role of ONNX

**ONNX (Open Neural Network Exchange)** is used to export our tracking math into a standardized computation graph.

- **Portability:** The Kalman Filter and CamShift lookup logic can be ported to **C++, Unity (C#), or Web** without re-writing the math.
- **Optimization:** ONNX Runtime allows us to leverage SIMD instructions (AVX2/AVX512) and specialized hardware accelerators.
- **Observation:** In our CPU-only benchmarks, NumPy is often faster for small 6x6 matrices due to the overhead of calling the ONNX Runtime engine. However, the ONNX infrastructure is ready for **GPU deployment (TensorRT)** where it can handle thousands of parallel tracks.

---

## 🖥️ Benchmark Environment
- **CPU:** AMD Ryzen 7 5800U (8 Cores, 16 Threads)
- **GPU:** AMD Radeon Graphics (Integrated) - *Tested on CPU path*
- **OS:** Windows 11
- **Optimization:** OpenCL enabled; XNNPACK Delegate for TFLite.

---

## 🛠️ How to Reproduce
1. Record a dataset: `python -m foodninja.benchmark --collect`
2. Run analysis: `python -m foodninja.benchmark --all`
3. View HTML Report: `benchmark_results/benchmark_report.html`

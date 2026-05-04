"""
runner.py
=========
CLI entry point for the Food Ninja benchmark suite.

Usage
-----
    # Run everything (recommended first time)
    python -m foodninja.benchmark --all

    # Individual suites
    python -m foodninja.benchmark --fps
    python -m foodninja.benchmark --latency
    python -m foodninja.benchmark --models           # requires downloaded variants
    python -m foodninja.benchmark --onnx

    # Collect webcam dataset for benchmarks
    python -m foodninja.benchmark --collect

    # Download missing MediaPipe model variants first
    python -m foodninja.benchmark --download-models

    # Choose output directory (default: benchmark_results/)
    python -m foodninja.benchmark --all --out benchmark_results

All JSON and chart outputs land in --out (default ``benchmark_results/``).
The HTML report is always written at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from foodninja.core.utils import get_resource_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON saved → {path}")


def _section(title: str) -> None:
    bar = "─" * (len(title) + 4)
    print(f"\n┌{bar}┐")
    print(f"│  {title}  │")
    print(f"└{bar}┘")


# ---------------------------------------------------------------------------
# Sub-command runners
# ---------------------------------------------------------------------------

def run_download(models_dir: Path) -> dict[str, Path]:
    from foodninja.benchmark.model_downloader import ensure_models
    _section("Downloading MediaPipe Model Variants")
    return ensure_models(models_dir)


def run_collect(dataset_path: Path) -> None:
    from foodninja.benchmark.data_collector import collect_dataset
    _section("Collecting Webcam Dataset")
    collect_dataset(dataset_path)


def run_fps(out_dir: Path, model_path: str | None, dataset_path: Path) -> None:
    from foodninja.benchmark import fps_bench
    from foodninja.benchmark.latency_bench import render_chart as _no_chart
    _section("FPS Benchmark")
    data = fps_bench.run(model_path=model_path, dataset_path=dataset_path)
    _save_json(data, out_dir / "benchmark_fps.json")


def run_latency(out_dir: Path, model_path: str | None, dataset_path: Path) -> None:
    from foodninja.benchmark import latency_bench
    _section("Per-Stage Latency Analysis")
    data = latency_bench.run(model_path=model_path, dataset_path=dataset_path)
    _save_json(data, out_dir / "benchmark_latency.json")
    print("  Rendering chart …")
    latency_bench.render_chart(data, out_dir / "latency_breakdown.png")


def run_models(out_dir: Path, models_dir: Path, available: dict[str, Path] | None, dataset_path: Path) -> None:
    from foodninja.benchmark import model_compare
    _section("Model Size vs Accuracy Tradeoff")
    data = model_compare.run(models_dir=models_dir, available_variants=available, dataset_path=dataset_path)
    _save_json(data, out_dir / "benchmark_models.json")
    print("  Rendering chart …")
    model_compare.render_chart(data, out_dir / "model_tradeoff.png")


def run_onnx(out_dir: Path) -> None:
    from foodninja.benchmark import onnx_export
    _section("ONNX Export & Runtime Benchmark")
    data = onnx_export.run(output_dir=out_dir)
    _save_json(data, out_dir / "benchmark_onnx.json")
    print("  Rendering chart …")
    onnx_export.render_chart(data, out_dir / "onnx_speedup.png")


def run_report(out_dir: Path) -> None:
    from foodninja.benchmark import report
    _section("Generating HTML Report")
    report.render_html(out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foodninja.benchmark",
        description="Food Ninja production CV benchmark suite",
    )
    parser.add_argument("--fps",              action="store_true", help="Run FPS benchmark")
    parser.add_argument("--latency",          action="store_true", help="Run per-stage latency analysis")
    parser.add_argument("--models",           action="store_true", help="Run model size vs accuracy comparison")
    parser.add_argument("--onnx",             action="store_true", help="Export ONNX graphs and benchmark")
    parser.add_argument("--collect",          action="store_true", help="Record webcam dataset to benchmark_results/dataset.mp4")
    parser.add_argument("--all",              action="store_true", help="Run all benchmarks")
    parser.add_argument("--download-models",  action="store_true", help="Download Lite/Heavy model variants")
    parser.add_argument("--out",              default="benchmark_results",
                        help="Output directory (default: benchmark_results/)")
    args = parser.parse_args(argv)

    if not any([args.fps, args.latency, args.models, args.onnx, args.all, args.download_models, args.collect]):
        parser.print_help()
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = out_dir / "dataset.mp4"

    models_dir = Path(get_resource_path("assets/models"))
    default_model = get_resource_path("assets/models/hand_landmarker.task")

    available_variants: dict[str, Path] | None = None

    t_start = time.perf_counter()

    # Download models if requested (or if --models implies it)
    if args.download_models or args.all or args.models:
        available_variants = run_download(models_dir)

    if args.collect:
        run_collect(dataset_path)

    if args.all or args.fps:
        run_fps(out_dir, model_path=default_model, dataset_path=dataset_path)

    if args.all or args.latency:
        run_latency(out_dir, model_path=default_model, dataset_path=dataset_path)

    if args.all or args.models:
        run_models(out_dir, models_dir=models_dir, available=available_variants, dataset_path=dataset_path)

    if args.all or args.onnx:
        run_onnx(out_dir)

    # Always regenerate HTML report
    run_report(out_dir)

    elapsed = time.time() - t_start
    print(f"\n[DONE] Benchmark suite finished in {elapsed:.1f}s")
    print(f"       Open: {out_dir / 'benchmark_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

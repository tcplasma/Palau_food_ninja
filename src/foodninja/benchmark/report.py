"""
report.py
=========
Aggregates all benchmark JSON results into a self-contained HTML report
with embedded dark-mode charts.

Usage (called by runner.py automatically):
    from foodninja.benchmark.report import render_html
    render_html(results_dir=Path("benchmark_results"))
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Food Ninja — CV Benchmark Report</title>
  <style>
    :root {{
      --bg:      #0f0f1a;
      --card:    #1a1a2e;
      --accent:  #4cc9f0;
      --accent2: #f77f00;
      --text:    #e0e0f0;
      --muted:   #888899;
      --border:  #2a2a4a;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem;
    }}
    h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: 0.25rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; font-size: 0.9rem; }}
    h2 {{
      font-size: 1.25rem; color: var(--accent2);
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.4rem; margin: 2rem 0 1rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.5rem;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
    }}
    th {{
      text-align: left; color: var(--muted);
      padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border);
      font-weight: 600;
    }}
    td {{
      padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border);
    }}
    tr:last-child td {{ border-bottom: none; }}
    .badge {{
      display: inline-block; padding: 0.15rem 0.5rem;
      border-radius: 4px; font-size: 0.75rem; font-weight: 700;
    }}
    .badge-green  {{ background: #1d4d2a; color: #6ee77a; }}
    .badge-yellow {{ background: #4d3d10; color: #f0c040; }}
    .badge-blue   {{ background: #0d2e4d; color: #4cc9f0; }}
    .badge-orange {{ background: #4d2600; color: #f77f00; }}
    img.chart {{
      width: 100%; max-width: 800px; border-radius: 8px;
      display: block; margin: 1rem auto;
    }}
    .gpu-note {{
      color: var(--muted); font-size: 0.82rem;
      border-left: 3px solid var(--accent2);
      padding-left: 0.75rem; margin-top: 0.75rem;
    }}
    footer {{
      color: var(--muted); font-size: 0.8rem;
      margin-top: 3rem; text-align: center;
    }}
  </style>
</head>
<body>
  <h1>🥷 Food Ninja — CV Benchmark Report</h1>
  <p class="subtitle">Generated {timestamp} &nbsp;·&nbsp; AMD Ryzen 7 5800U (CPU-only, AMD Radeon integrated GPU)</p>

  {fps_section}
  {latency_section}
  {models_section}
  {onnx_section}
  {help_section}

  <footer>Food Ninja Production CV Benchmark &copy; {year}</footer>
</body>
</html>
"""


def _img_tag(png_path: Path) -> str:
    if not png_path.exists():
        return "<p style='color:#888'>Chart not generated.</p>"
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f'<img class="chart" src="data:image/png;base64,{b64}" alt="{png_path.stem}"/>'


def _badge(text: str, colour: str = "blue") -> str:
    return f'<span class="badge badge-{colour}">{text}</span>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _fps_section(data: dict, charts_dir: Path) -> str:
    gpu = data.get("gpu", {})
    results = data.get("results", [])

    rows = ""
    for r in results:
        if "error" in r:
            rows += f"<tr><td>{r.get('path','?')}</td><td colspan='6' style='color:#e05050'>{r['error']}</td></tr>"
            continue
        fps = r.get("mean_fps", 0)
        colour = "green" if fps >= 60 else ("yellow" if fps >= 30 else "orange")
        rows += (
            f"<tr>"
            f"<td>{r.get('path','')}</td>"
            f"<td>{_badge(f'{fps:.0f} FPS', colour)}</td>"
            f"<td>{r.get('p50_ms','–')} ms</td>"
            f"<td>{r.get('p95_ms','–')} ms</td>"
            f"<td>{r.get('p99_ms','–')} ms</td>"
            f"<td>{r.get('std_ms','–')} ms</td>"
            f"</tr>"
        )

    gpu_note = ""
    if gpu.get("note"):
        gpu_note = f'<p class="gpu-note">⚠ {gpu["note"]}</p>'

    chart = _img_tag(charts_dir / "latency_breakdown.png")  # reuse if no fps chart
    return f"""
  <h2>⚡ FPS Benchmark — CPU vs GPU</h2>
  <div class="card">
    <table>
      <thead><tr>
        <th>Path</th><th>Mean FPS</th><th>p50</th><th>p95</th><th>p99</th><th>Std</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    {gpu_note}
  </div>
"""


def _latency_section(data: dict, charts_dir: Path) -> str:
    stages = data.get("stages", [])
    rows = ""
    for s in stages:
        mean = s.get("mean_us", "–")
        p95  = s.get("p95_us", "–")
        colour = "green" if isinstance(mean, float) and mean < 500 else "yellow"
        rows += (
            f"<tr>"
            f"<td>{s.get('stage','')}</td>"
            f"<td>{_badge(f'{mean} µs', colour) if mean != '–' else '–'}</td>"
            f"<td>{p95} µs</td>"
            f"<td>{s.get('p99_us','–')} µs</td>"
            f"<td>{s.get('std_us','–')} µs</td>"
            f"<td>{s.get('n_samples','–')}</td>"
            f"</tr>"
        )

    chart = _img_tag(charts_dir / "latency_breakdown.png")
    return f"""
  <h2>⏱ Per-Stage Latency Analysis</h2>
  <div class="card">
    {chart}
    <table>
      <thead><tr>
        <th>Stage</th><th>Mean</th><th>p95</th><th>p99</th><th>Std</th><th>Samples</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
"""


def _models_section(data: dict, charts_dir: Path) -> str:
    variants = data.get("variants", [])
    rows = ""
    for v in variants:
        if "error" in v:
            rows += f"<tr><td>{v.get('variant','?')}</td><td colspan='5' style='color:#e05050'>{v['error']}</td></tr>"
            continue
        rate = v.get("detection_rate", 0)
        colour = "green" if rate > 0.8 else ("yellow" if rate > 0.5 else "orange")
        rows += (
            f"<tr>"
            f"<td>{_badge(v.get('variant','').capitalize(), 'blue')}</td>"
            f"<td>{v.get('model_size_mb','–')} MB</td>"
            f"<td>{v.get('inference_mean_ms','–')} ms</td>"
            f"<td>{v.get('inference_p95_ms','–')} ms</td>"
            f"<td>{_badge(f'{rate*100:.0f}%', colour)}</td>"
            f"<td>{v.get('bbox_stability_sigma_px','–')} px σ</td>"
            f"</tr>"
        )

    chart = _img_tag(charts_dir / "model_tradeoff.png")
    return f"""
  <h2>📊 Model Size vs Accuracy Tradeoff</h2>
  <div class="card">
    {chart}
    <table>
      <thead><tr>
        <th>Variant</th><th>Size</th><th>Latency (mean)</th><th>Latency (p95)</th>
        <th>Detect Rate</th><th>BBox Stability</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
"""


def _onnx_section(data: dict, charts_dir: Path) -> str:
    graphs = data.get("graphs", [])
    rows = ""
    for g in graphs:
        if "error" in g:
            rows += f"<tr><td>{g.get('graph','?')}</td><td colspan='5' style='color:#e05050'>{g['error']}</td></tr>"
            continue
        speedup = g.get("speedup_x")
        colour = "green" if speedup and speedup > 1 else "yellow"
        rows += (
            f"<tr>"
            f"<td>{g.get('graph','')}</td>"
            f"<td>{g.get('model_size_kb','–')} KB</td>"
            f"<td>{round(g.get('numpy_mean_ms',0)*1000,1)} µs</td>"
            f"<td>{round(g.get('onnx_mean_ms',0)*1000,1)} µs</td>"
            f"<td>{_badge(f'×{speedup}', colour) if speedup else '–'}</td>"
            f"<td>{g.get('provider','CPUExecutionProvider')}</td>"
            f"</tr>"
        )

    chart = _img_tag(charts_dir / "onnx_speedup.png")
    return f"""
  <h2>🔷 ONNX Export — Tracking Core Ops</h2>
  <div class="card">
    {chart}
    <table>
      <thead><tr>
        <th>Graph</th><th>ONNX Size</th><th>NumPy</th><th>ONNX Runtime</th>
        <th>Speedup</th><th>Provider</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="gpu-note">
      ONNX graphs exported: <code>kalman_predict.onnx</code> (F·x and F·P·F^T + Q) and
      <code>camshift_lookup.onnx</code> (H-S histogram back-projection via Gather).
      MediaPipe hand-landmarker models are TFLite-based; full TFLite→ONNX conversion
      requires tf2onnx and is outside this benchmark scope.
    </p>
  </div>
"""


def _help_section() -> str:
    return """
  <h2>📖 Understanding the Metrics</h2>
  <div class="card">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
      <div>
        <h3 style="color: var(--accent); margin-bottom: 0.5rem; font-size: 1rem;">⚡ FPS & Latency</h3>
        <p style="font-size: 0.82rem; margin-bottom: 0.5rem; color: var(--muted);">
          <strong>FPS (Frames Per Second):</strong> Measures total throughput. 60+ FPS is the production gold standard for fluid interaction.
        </p>
        <p style="font-size: 0.82rem; color: var(--muted);">
          <strong>Latency (µs):</strong> The time cost of individual pipeline stages. 1ms = 1000µs. Lower latency reduces input lag, making tracking feel "snappy".
        </p>
      </div>
      <div>
        <h3 style="color: var(--accent); margin-bottom: 0.5rem; font-size: 1rem;">📊 Models & ONNX</h3>
        <p style="font-size: 0.82rem; margin-bottom: 0.5rem; color: var(--muted);">
          <strong>Detection Rate:</strong> Reliability of MediaPipe hand detection. 90%+ is required for a robust production user experience.
        </p>
        <p style="font-size: 0.82rem; color: var(--muted);">
          <strong>ONNX Speedup:</strong> Ratio of NumPy vs ONNX Runtime. For small 6x6 matrices (Kalman), NumPy is often faster due to lower call overhead on CPU.
        </p>
      </div>
    </div>
  </div>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_html(results_dir: Path) -> Path:
    """Read all benchmark JSON files and write benchmark_report.html."""
    results_dir = Path(results_dir)

    def _load(name: str) -> dict:
        p = results_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    fps_data     = _load("benchmark_fps.json")
    latency_data = _load("benchmark_latency.json")
    models_data  = _load("benchmark_models.json")
    onnx_data    = _load("benchmark_onnx.json")

    now = datetime.now()
    html = _HTML_TEMPLATE.format(
        timestamp=now.strftime("%Y-%m-%d %H:%M"),
        year=now.year,
        fps_section=_fps_section(fps_data, results_dir),
        latency_section=_latency_section(latency_data, results_dir),
        models_section=_models_section(models_data, results_dir),
        onnx_section=_onnx_section(onnx_data, results_dir),
        help_section=_help_section(),
    )

    out_path = results_dir / "benchmark_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Report saved → {out_path}")
    return out_path

"""
model_downloader.py
====================
Auto-downloads the three MediaPipe hand-landmarker model variants
(Lite / Full / Heavy) from Google's CDN if they are not already present
in ``assets/models/``.

MediaPipe model registry:
  https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

File sizes (approximate):
  hand_landmarker_lite.task    ~1.0 MB
  hand_landmarker.task         ~8.3 MB   ← already shipped with the project
  hand_landmarker_heavy.task   ~25  MB
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

_CDN_BASE = "https://storage.googleapis.com/mediapipe-models/hand_landmarker"

MODELS: dict[str, dict] = {
    "lite": {
        "filename": "hand_landmarker_lite.task",
        "url": f"{_CDN_BASE}/hand_landmarker/lite/1/0/hand_landmarker.task",
        "description": "Lite (~1 MB) — fastest, best for edge devices",
    },
    "full": {
        "filename": "hand_landmarker.task",
        "url": f"{_CDN_BASE}/hand_landmarker/full/1/0/hand_landmarker.task",
        "description": "Full (~8.3 MB) — default, balanced accuracy/speed",
    },
    "heavy": {
        "filename": "hand_landmarker_heavy.task",
        "url": f"{_CDN_BASE}/hand_landmarker/heavy/1/0/hand_landmarker.task",
        "description": "Heavy (~25 MB) — highest accuracy, slowest",
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded / total_size * 100
    bar = "#" * int(pct // 2)
    print(f"\r  [{bar:<50}] {pct:5.1f}%  ({downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB)", end="", flush=True)


def ensure_models(models_dir: Path, variants: list[str] | None = None) -> dict[str, Path]:
    """
    Ensure the requested model variants exist in *models_dir*.

    Parameters
    ----------
    models_dir:
        Destination directory (created if it does not exist).
    variants:
        Subset of ``['lite', 'full', 'heavy']`` to download.
        ``None`` means all three.

    Returns
    -------
    dict mapping variant name → resolved Path of the ``.task`` file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variants = list(MODELS.keys())

    result: dict[str, Path] = {}

    for variant in variants:
        info = MODELS[variant]
        dest = models_dir / info["filename"]

        if dest.exists():
            print(f"  [OK] {variant:6s}  {info['filename']}  (already present, {dest.stat().st_size/1024/1024:.1f} MB)")
            result[variant] = dest
            continue

        print(f"  [DL] {variant:6s}  {info['filename']}  ({info['description']})")
        try:
            urllib.request.urlretrieve(info["url"], dest, reporthook=_progress_hook)
            print()  # newline after progress bar
            print(f"    Saved -> {dest}  ({dest.stat().st_size/1024/1024:.1f} MB)")
            result[variant] = dest
        except Exception as exc:
            print(f"\n  [ERR] Failed to download {variant}: {exc}")
            # Don't abort — just skip this variant

    return result


if __name__ == "__main__":
    import sys
    from foodninja.core.utils import get_resource_path

    models_dir = Path(get_resource_path("assets/models"))
    print("Downloading MediaPipe hand-landmarker model variants …\n")
    paths = ensure_models(models_dir)
    print("\nDone.")
    for name, p in paths.items():
        print(f"  {name:6s} → {p}")

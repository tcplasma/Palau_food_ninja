# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Dynamically find mediapipe path
try:
    import mediapipe
    mediapipe_path = os.path.dirname(mediapipe.__file__)
except ImportError:
    mediapipe_path = ""

# Define assets to include
added_files = [
    ('assets', 'assets'),
    ('assets/food_catalog.json', 'assets'),
    ('assets/trap_catalog.json', 'assets'),
]

# Specifically add mediapipe data if found
if mediapipe_path:
    # On some systems, collect_data_files('mediapipe') is enough
    # but manually adding the whole directory ensures all .binarypb models are there
    added_files.append((mediapipe_path, 'mediapipe'))

a = Analysis(
    ['src/foodninja/app.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=['cv2', 'numpy', 'pygame', 'mediapipe'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NutritionNinja',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Set to False for windowed mode in production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NutritionNinja',
)

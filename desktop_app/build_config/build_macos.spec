# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for macOS build
"""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path('.').resolve().parent

block_cipher = None

a = Analysis(
    ['../main.py'],
    pathex=[str(project_root / 'desktop_app')],
    binaries=[],
    datas=[
        ('../resources/logo.png', 'resources'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'pydantic',
        'pydantic_settings',
        'requests',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
        'cv2',
        'mediapipe',
        'ultralytics',
    ],
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
    name='LocopilotCVVR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='LocopilotCVVR',
)

app = BUNDLE(
    coll,
    name='LocopilotCVVR.app',
    icon=None,
    bundle_identifier='com.mindcoin.locopilot-cvvr',
    version='1.0.0',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'LSBackgroundOnly': 'False',
        'CFBundleName': 'Locopilot CVVR',
        'CFBundleDisplayName': 'Locopilot CVVR',
        'CFBundleIdentifier': 'com.mindcoin.locopilot-cvvr',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSRequiresAquaSystemAppearance': False,
    },
)


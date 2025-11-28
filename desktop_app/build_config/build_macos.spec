# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for macOS build
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# Get the project root directory
project_root = Path('.').resolve().parent

block_cipher = None

# Build the datas list conditionally
datas_list = [
    ('../resources/logo.png', 'desktop_app/resources'),
    ('../__init__.py', 'desktop_app'),
    ('../main.py', 'desktop_app'),
    ('../models', 'desktop_app/models'),
    ('../views', 'desktop_app/views'),
    ('../controllers', 'desktop_app/controllers'),
    ('../services', 'desktop_app/services'),
    ('../utils', 'desktop_app/utils'),
    ('../../app', 'app'),  # Include FastAPI backend code
    ('../../requirements.txt', '.'),  # Backend requirements reference
]

# Add certifi data files for SSL certificate handling
datas_list += collect_data_files('certifi')

# Conditionally add YOLO model if it exists
yolo_model_path = Path('../../yolo11s.pt')
if yolo_model_path.exists():
    datas_list.append(('../../yolo11s.pt', '.'))

a = Analysis(
    ['../launcher.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas_list,
    hiddenimports=[
        # Desktop app imports
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'pydantic',
        'pydantic_settings',
        'requests',
        
        # Backend - Uvicorn/ASGI imports
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        
        # Backend - FastAPI/Starlette imports
        'fastapi',
        'fastapi.routing',
        'fastapi.responses',
        'fastapi.middleware',
        'fastapi.middleware.cors',
        'fastapi.exceptions',
        'fastapi.background',
        'fastapi.datastructures',
        'starlette',
        'starlette.routing',
        'starlette.middleware',
        'starlette.middleware.base',
        'starlette.middleware.cors',
        'starlette.applications',
        'starlette.exceptions',
        'starlette.requests',
        'starlette.responses',
        
        # Backend - ML/CV imports
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        
        # Backend - Multiprocessing
        'multiprocessing',
        'multiprocessing.pool',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'mediapipe',
        # Note: numpy, PIL, cv2, ultralytics are now INCLUDED for backend
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


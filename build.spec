# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all necessary data files and submodules
datas = []
hiddenimports = []

# Add ultralytics data files
datas += collect_data_files('ultralytics')

# Add app directory
datas += [('app', 'app')]
datas += [('models', 'models')]
datas += [('assets', 'assets')]
datas += [('config.json', '.')]

# Collect all submodules for critical packages
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('lapx')
hiddenimports += collect_submodules('fastdtw')

# Additional hidden imports
hiddenimports += [
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
]

# Get torch library path
import torch
torch_path = os.path.dirname(torch.__file__)

# Collect all torch DLLs and libraries - be more comprehensive
binaries = []
if sys.platform == 'win32':
    # Add torch lib directory - collect ALL files, not just DLLs
    torch_lib_path = os.path.join(torch_path, 'lib')
    if os.path.exists(torch_lib_path):
        for file in os.listdir(torch_lib_path):
            # Include DLLs and other binary files
            if file.endswith(('.dll', '.pyd', '.so')):
                binaries.append((os.path.join(torch_lib_path, file), 'torch/lib'))
                # CRITICAL FIX: Also copy important DLLs to root directory
                # This helps with DLL dependency resolution
                if any(x in file.lower() for x in ['libiomp5md', 'libomp', 'msvcp', 'vcruntime', 'c10', 'torch_cpu']):
                    binaries.append((os.path.join(torch_lib_path, file), '.'))
    
    # Try to include Visual C++ redistributable DLLs from system
    import ctypes.util
    vcruntime_dlls = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll']
    for dll_name in vcruntime_dlls:
        dll_path = ctypes.util.find_library(dll_name.replace('.dll', ''))
        if dll_path and os.path.exists(dll_path):
            binaries.append((dll_path, '.'))

# Create runtime hook to help with DLL loading
runtime_hook_content = '''
import os
import sys

# CRITICAL: Set up DLL paths BEFORE any other imports
if hasattr(sys, '_MEIPASS'):
    # Add base directory to DLL search path first
    try:
        os.add_dll_directory(sys._MEIPASS)
    except (OSError, AttributeError):
        pass
    
    # Add torch lib to DLL search path
    torch_lib = os.path.join(sys._MEIPASS, 'torch', 'lib')
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except (OSError, AttributeError):
            pass
        # Also add to PATH at the beginning
        os.environ['PATH'] = sys._MEIPASS + os.pathsep + torch_lib + os.pathsep + os.environ.get('PATH', '')
    
    # Set environment variables for PyTorch
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''

# Write runtime hook
runtime_hook_path = 'pyi_rth_torch.py'
with open(runtime_hook_path, 'w') as f:
    f.write(runtime_hook_content)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[runtime_hook_path],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Ensure all torch binaries are included
for item in a.binaries:
    if 'torch' in item[0].lower():
        print(f"Included torch binary: {item[0]}")


pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoloV8_TD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YoloV8_TD',
)

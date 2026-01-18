
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

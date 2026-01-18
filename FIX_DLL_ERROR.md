# Sửa Lỗi PyInstaller DLL (WinError 1114)

## Tóm Tắt Lỗi
Lỗi `OSError: [WinError 1114]` xảy ra khi PyTorch không thể khởi tạo DLL `c10.dll` trong ứng dụng đã build bằng PyInstaller. Đây là lỗi phổ biến do thiếu dependencies hoặc đường dẫn DLL không đúng.

## Các Cải Tiến Đã Thực Hiện

### 1. Cải Thiện Bundling DLL trong `build.spec`

#### a) Bao Gồm Tất Cả DLL Quan Trọng
```python
# Copy các DLL quan trọng vào cả thư mục root và torch/lib
if any(x in file.lower() for x in ['libiomp5md', 'libomp', 'msvcp', 'vcruntime', 'c10', 'torch_cpu']):
    binaries.append((os.path.join(torch_lib_path, file), '.'))
```

#### b) Thêm Visual C++ Runtime DLLs
```python
# Tự động tìm và bao gồm Visual C++ redistributable DLLs
vcruntime_dlls = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll']
for dll_name in vcruntime_dlls:
    dll_path = ctypes.util.find_library(dll_name.replace('.dll', ''))
    if dll_path and os.path.exists(dll_path):
        binaries.append((dll_path, '.'))
```

### 2. Runtime Hook Cải Tiến

Runtime hook mới thiết lập đường dẫn DLL **TRƯỚC KHI** import bất kỳ module nào:

```python
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
```

### 3. Kiểm Tra Dependencies

File `test_dll_dependencies.py` đã được tạo để kiểm tra:
- Tất cả các DLL trong thư mục torch/lib
- Visual C++ Runtime dependencies
- Khả năng import PyTorch

## Cách Build Lại

### Bước 1: Làm Sạch Build Cũ
```powershell
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
```

### Bước 2: Kiểm Tra Dependencies (Optional)
```powershell
python test_dll_dependencies.py
```

### Bước 3: Build Với PyInstaller
```powershell
pyinstaller build.spec --clean
```

### Bước 4: Test Ứng Dụng
```powershell
cd dist\YoloV8_TD
.\YoloV8_TD.exe
```

## Nếu Vẫn Còn Lỗi

### Giải Pháp 1: Cài Đặt Visual C++ Redistributable
Tải và cài đặt từ Microsoft:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Giải Pháp 2: Sử dụng PyTorch CPU-Only Build
Nếu đang dùng GPU version, chuyển sang CPU version:
```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Giải Pháp 3: Kiểm Tra DLL Trong Dist
Sau khi build, kiểm tra xem các DLL có trong dist không:
```powershell
dir "dist\YoloV8_TD\_internal" -Recurse -Filter "*.dll" | Where-Object {$_.Name -like "*c10*" -or $_.Name -like "*torch*"}
```

Các DLL quan trọng cần có:
- `c10.dll`
- `torch_cpu.dll`
- `torch_python.dll`
- `libiomp5md.dll`
- `vcruntime140.dll`
- `msvcp140.dll`

### Giải Pháp 4: Dependency Walker (Nâng Cao)
Sử dụng Dependency Walker để xem DLL nào thiếu:
1. Tải Dependency Walker: https://www.dependencywalker.com/
2. Mở `c10.dll` từ `dist\YoloV8_TD\_internal\torch\lib\c10.dll`
3. Xem các DLL màu đỏ (thiếu)

### Giải Pháp 5: Debug Mode
Bật debug mode trong `build.spec`:
```python
exe = EXE(
    ...
    debug=True,  # Changed from False
    console=True,
    ...
)
```

Khi chạy, sẽ thấy thông tin chi tiết hơn về lỗi.

## Thông Tin Hệ Thống

- **PyTorch Version:** 2.9.1+cpu
- **PyInstaller:** 6.9.0
- **Python:** 3.10.5
- **Platform:** Windows 10

## Các DLL Trong PyTorch Lib

```
c10.dll (1.04 MB)
libiomp5md.dll (1.54 MB)
libiompstubs5md.dll (0.04 MB)
shm.dll (0.01 MB)
torch.dll (0.01 MB)
torch_cpu.dll (250.49 MB)
torch_global_deps.dll (0.01 MB)
torch_python.dll (17.45 MB)
uv.dll (0.19 MB)
```

## Lưu Ý

1. **Build mất thời gian:** PyTorch rất lớn (~250MB chỉ riêng torch_cpu.dll), build có thể mất 5-15 phút
2. **UPX compression:** Đã bật UPX để giảm kích thước file
3. **Console mode:** Đang để console=True để debug, có thể chuyển sang False sau khi đã hoạt động ổn định
4. **Clean build:** Luôn clean build cũ trước khi build lại để tránh cache cũ

## Liên Hệ Nếu Cần Hỗ Trợ Thêm

Nếu vẫn gặp lỗi sau khi thử các giải pháp trên, vui lòng cung cấp:
1. Full error traceback
2. Output của `test_dll_dependencies.py`
3. Danh sách DLL trong `dist\YoloV8_TD\_internal`

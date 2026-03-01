# HƯỚNG DẪN SỬ DỤNG FILE EXE

## ✅ ĐÃ ĐÓNG GÓI THÀNH CÔNG!

File ứng dụng đã được tạo tại:
```
D:\Yolov8 TD - Main\dist\YoloV8_TD\
```

### 📁 Cấu trúc thư mục
Toàn bộ thư mục `YoloV8_TD` chứa:
- **YoloV8_TD.exe** - File chương trình chính (click đúp để chạy)
- Các file DLL và thư viện cần thiết
- Thư mục app, models, assets, config.json

### 🚀 Cách chạy ứng dụng

1. **Chạy trực tiếp:**
   - Vào thư mục `D:\Yolov8 TD - Main\dist\YoloV8_TD\`
   - Double-click vào file `YoloV8_TD.exe`

2. **Tạo shortcut:**
   - Click phải vào `YoloV8_TD.exe`
   - Chọn "Create shortcut"
   - Di chuyển shortcut ra Desktop hoặc nơi thuận tiện

### 📦 Phân phối ứng dụng

Nếu bạn muốn chia sẻ ứng dụng cho người khác:

**Cách 1: Chia sẻ toàn bộ thư mục**
- Copy toàn bộ thư mục `YoloV8_TD` 
- Nén thành file ZIP nếu cần
- Người nhận giải nén và chạy file .exe

**Cách 2: Tạo installer (tùy chọn)**
- Sử dụng Inno Setup hoặc NSIS để tạo file cài đặt
- Tự động tạo shortcuts và cài đặt vào Program Files

### ⚠️ Lưu ý quan trọng

1. **Antivirus có thể cảnh báo:**
   - Windows Defender hoặc antivirus khác có thể chặn file .exe
   - Điều này là bình thường với các file .exe được tạo bởi PyInstaller
   - Cách khắc phục: Thêm file vào whitelist/exception của antivirus

2. **Không xóa các file đi kèm:**
   - File .exe cần tất cả các file DLL và thư mục bên cạnh để hoạt động
   - KHÔNG chỉ copy file .exe ra ngoài, sẽ không chạy được

3. **Kích thước:**
   - Toàn bộ thư mục khá lớn (~1-2GB) do chứa PyTorch và YOLOv8
   - Đây là điều bình thường cho ứng dụng AI/Deep Learning

---

**Tác giả:** Build bởi Vo Kiet NGT
**Ngày tạo:** 14/01/2026
**Framework:** YOLOv8, PyTorch, TKinter

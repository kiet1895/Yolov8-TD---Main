import numpy as np
import os
import sys

def get_resource_path(relative_path):
    """
    Lấy đường dẫn tuyệt đối đến tài nguyên, hoạt động cả trong môi trường dev và PyInstaller.
    
    Khi chạy từ .exe (PyInstaller):
    - Các file nội bộ (models, app modules) nằm trong _MEIPASS (_internal)
    - Các file bên ngoài (assets, config.json) nằm cùng thư mục với .exe
    """
    # Xác định thư mục gốc
    if getattr(sys, 'frozen', False):
        # Chạy từ exe - sử dụng thư mục chứa exe
        # sys.executable = đường dẫn đến file .exe
        # os.path.dirname(sys.executable) = thư mục chứa .exe
        base_path = os.path.dirname(sys.executable)
    else:
        # Môi trường development
        base_path = os.path.abspath(".")
    
    # Chuẩn hóa đường dẫn tương đối và ghép với base_path
    full_path = os.path.normpath(os.path.join(base_path, relative_path))
    
    # Nếu đang chạy từ exe và file không tồn tại ở exe's directory,
    # thử tìm trong _MEIPASS (cho các file được đóng gói như models)
    if getattr(sys, 'frozen', False) and not os.path.exists(full_path):
        try:
            internal_path = os.path.normpath(os.path.join(sys._MEIPASS, relative_path))
            if os.path.exists(internal_path):
                return internal_path
        except AttributeError:
            pass
    
    return full_path

def calculate_angle(a, b, c):
    """
    Tính góc tạo bởi 3 điểm a, b, c với góc tại điểm b.
    Các điểm được cho dưới dạng list hoặc tuple (x, y).
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_midpoint(p1, p2):
    """
    Tính điểm trung tâm giữa hai điểm p1 và p2.
    """
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
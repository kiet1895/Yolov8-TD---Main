import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import threading
import queue
import numpy as np
import sys

import webbrowser
import requests
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
from . import assessment
from .utils import get_resource_path
import shutil

def get_config_path():
    """
    Get the path to config.json with smart handling for bundled exe.
    
    Strategy:
    - In development: use config.json in current directory
    - In bundled exe: use config.json in exe's directory (not _internal)
    - If config doesn't exist in exe's directory but exists in _internal, copy it out
    
    This ensures config changes persist across rebuilds.
    """
    if getattr(sys, 'frozen', False):
        # Running from bundled exe
        exe_dir = os.path.dirname(sys.executable)
        user_config = os.path.join(exe_dir, 'config.json')
        
        # If user config doesn't exist, try to copy from _internal (first run)
        if not os.path.exists(user_config):
            internal_config = get_resource_path('config.json')
            if os.path.exists(internal_config):
                try:
                    shutil.copy2(internal_config, user_config)
                    print(f"Đã copy config từ _internal ra {user_config}")
                except Exception as e:
                    print(f"Không thể copy config: {e}")
        
        return user_config
    else:
        # Development mode - use local config.json
        return os.path.abspath('config.json')

CONFIG_FILE = get_config_path()

# --- Cấu hình cho các tính năng mới ---
CURRENT_VERSION = "1.0.0"
UPDATE_CHECK_URL = "https://api.github.com/repos/TEN_DANG_NHAP_CUA_BAN/TEN_KHO_LUU_TRU/releases/latest" # THAY ĐỔI: URL để kiểm tra phiên bản mới
RESULTS_API_ENDPOINT = "http://your-server.com/api/submit_results" # TÙY CHỌN: URL để gửi kết quả nếu bạn có server riêng

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần mềm Hỗ trợ Giáo dục Thể chất")
        self.geometry("600x400")

        self.teacher_logged_in = False
        self.app_config = self.load_config()

        self.create_main_widgets()
        self.populate_exercises()

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            messagebox.showinfo("Thông báo", f"Không tìm thấy file '{CONFIG_FILE}'.\nSẽ tạo một file cấu hình mặc định.")
            default_config = {
                "teacher_credentials": {
                    "username": "teacher",
                    "password": "123"
                },
                "exercises": [
                    {
                        "name": "Bài tập vươn thở (Mẫu)",
                        "standard_video": "video-mau-chuan.mp4",
                        "cache_file": "cache_vuon_tho.json",
                        "frame_mapping": {
                            "Nhịp 1": [30, 60],
                            "Nhịp 2": [60, 90],
                            "Nhịp 3": [90, 120],
                            "Nhịp 4": [120, 180],
                            "Nhịp 5": [180, 210],
                            "Nhịp 6": [210, 270],
                            "Nhịp 7": [270, 300],
                            "Nhịp 8": [300, 390]
                        },
                        "phase_weights": {
                            "Nhịp 1": {
                                "L_SHOULDER_ANGLE": 0.5,
                                "R_SHOULDER_ANGLE": 0.5
                            },
                            "Nhịp 2": {
                                "L_SHOULDER_ANGLE": 0.3,
                                "R_SHOULDER_ANGLE": 0.3,
                                "L_HIP_ANGLE": 0.2,
                                "R_HIP_ANGLE": 0.2
                            },
                            "Nhịp 3": {
                                "R_ELBOW_ANGLE": 0.4,
                                "L_ELBOW_ANGLE": 0.4,
                                "WRIST_DISTANCE": 0.2
                            },
                            "Nhịp 4": {
                                "L_SHOULDER_ANGLE": 0.5,
                                "R_SHOULDER_ANGLE": 0.5
                            },
                            "Nhịp 5": {
                                "L_SHOULDER_ANGLE": 0.5,
                                "R_SHOULDER_ANGLE": 0.5
                            },
                            "Nhịp 6": {
                                "L_SHOULDER_ANGLE": 0.3,
                                "R_SHOULDER_ANGLE": 0.3,
                                "L_HIP_ANGLE": 0.2,
                                "R_HIP_ANGLE": 0.2
                            },
                            "Nhịp 7": {
                                "R_ELBOW_ANGLE": 0.4,
                                "L_ELBOW_ANGLE": 0.4,
                                "WRIST_DISTANCE": 0.2
                            },
                            "Nhịp 8": {
                                "L_SHOULDER_ANGLE": 0.5,
                                "R_SHOULDER_ANGLE": 0.5
                            }
                        }
                    }
                ]
            }
            self.app_config = default_config
            self.save_config()
            return default_config
        
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            messagebox.showerror("Lỗi", f"File cấu hình '{CONFIG_FILE}' bị lỗi hoặc trống.\nVui lòng xóa file này và chạy lại chương trình để tạo file mới.")
            self.destroy()
            return None

    def save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.app_config, f, indent=4, ensure_ascii=False)

    def create_main_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill="both")

        # --- Student Section ---
        student_frame = ttk.LabelFrame(main_frame, text="Dành cho Học sinh", padding="10")
        student_frame.pack(fill="x", pady=10)

        ttk.Label(student_frame, text="Chọn bài tập:").pack(pady=5)
        self.exercise_combo = ttk.Combobox(student_frame, state="readonly", width=40)
        self.exercise_combo.pack(pady=5)

        ttk.Button(student_frame, text="Chọn video và Bắt đầu Đánh giá", command=self.start_student_assessment).pack(pady=10)

        # --- Teacher Section ---
        teacher_frame = ttk.LabelFrame(main_frame, text="Dành cho Giáo viên", padding="10")
        teacher_frame.pack(fill="x", pady=10)

        self.login_button = ttk.Button(teacher_frame, text="Đăng nhập", command=self.open_login_window)
        self.login_button.pack(side="left", padx=5)

        self.admin_button = ttk.Button(teacher_frame, text="Quản lý Bài tập", state="disabled", command=self.open_teacher_admin)
        self.admin_button.pack(side="left", padx=5)

        # --- Update Section ---
        update_frame = ttk.Frame(main_frame)
        update_frame.pack(side="bottom", fill="x", pady=5)
        self.update_button = ttk.Button(update_frame, text="Kiểm tra cập nhật", command=self.check_for_updates)
        self.update_button.pack()

    def populate_exercises(self):
        if self.app_config:
            exercise_names = [ex['name'] for ex in self.app_config.get('exercises', [])]
            self.exercise_combo['values'] = exercise_names
            if exercise_names:
                self.exercise_combo.current(0)

    def start_student_assessment(self):
        selected_name = self.exercise_combo.get()
        if not selected_name:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn một bài tập để bắt đầu.")
            return

        exercise = next((ex for ex in self.app_config['exercises'] if ex['name'] == selected_name), None)
        if not exercise:
            messagebox.showerror("Lỗi", "Không tìm thấy thông tin cho bài tập đã chọn.")
            return

        student_video_path = filedialog.askopenfilename(title="Chọn video của bạn", filetypes=[("Video files", "*.mp4 *.avi")])
        if not student_video_path:
            return

        # messagebox.showinfo("Bắt đầu", "Chuẩn bị xử lý video mẫu (nếu cần). Quá trình này có thể mất vài phút.")
        std_data = assessment.load_or_extract_standard(exercise['standard_video'], exercise['cache_file'])
        if not std_data:
            messagebox.showerror("Lỗi Video Mẫu", f"Không thể xử lý video mẫu cho bài tập '{selected_name}'.")
            return
        # Cải tiến: Tính toán thông số chuẩn cho từng nhịp để tăng độ chính xác
        std_params_per_phase = assessment.calculate_standard_params_per_phase(std_data, exercise['frame_mapping'])
        # messagebox.showinfo("Sẵn sàng", "Dữ liệu chuẩn đã sẵn sàng. Bắt đầu đánh giá video của bạn (quá trình này có thể mất một lúc).")

        # --- Giai đoạn 1: Hiển thị video xử lý trực tiếp cho người dùng xem ---
        # messagebox.showinfo("Xem trước", "Cửa sổ xem trước video đang xử lý sẽ hiện lên.\nQuá trình tính điểm sẽ bắt đầu sau khi video kết thúc.")
        preview_window = AssessmentWindow(self, student_video_path, exercise)
        self.wait_window(preview_window) # Chờ cho cửa sổ xem trước đóng lại

        # --- Giai đoạn 2: Chạy phân tích nền với cửa sổ tiến trình ---
        progress_win = ProgressWindow(self, "Đang phân tích và tính điểm")
        self.result_queue = queue.Queue()

        # Tạo và bắt đầu luồng xử lý
        thread = threading.Thread(
            target=self.run_assessment_thread,
            args=(
                student_video_path, std_data, std_params_per_phase,
                exercise['frame_mapping'], exercise['phase_weights'],
                lambda p, s: self.after(0, progress_win.update_progress, p, s)
            )
        )
        thread.start()

        # Kiểm tra luồng xử lý định kỳ
        self.check_thread(thread, progress_win, selected_name)

    def show_final_results(self, phase_max_scores, exercise_name):
        if phase_max_scores is None: return

        scores = list(phase_max_scores.values())
        avg_score = (sum(scores) / len(scores)) if scores else 0.0

        result_str = "KẾT QUẢ ĐÁNH GIÁ CHI TIẾT\n" + "="*30 + "\n"
        for phase, score in phase_max_scores.items():
            # Căn chỉnh văn bản để dễ đọc hơn
            result_str += f"- {phase:<15}: {score*100:>5.1f}%\n"
        result_str += "-"*30 + f"\nTRUNG BÌNH CHUNG: {avg_score*100:.1f}%\n" + "-"*30 + "\n"

        # --- Cải tiến logic nhận xét ---
        # Tìm tất cả các nhịp có điểm dưới 70%
        LOW_SCORE_THRESHOLD = 0.7
        low_scoring_phases = {phase: score for phase, score in phase_max_scores.items() if score < LOW_SCORE_THRESHOLD}
        
        # Sắp xếp các nhịp yếu theo điểm từ thấp đến cao
        sorted_low_phases = sorted(low_scoring_phases.items(), key=lambda item: item[1])

        if avg_score >= 0.85 and not sorted_low_phases:
            result_str += ">> NHẬN XÉT: Em thực hiện rất tốt động tác!"
        elif not sorted_low_phases:
            result_str += ">> NHẬN XÉT: Em thực hiện tốt, các nhịp đều đạt yêu cầu."
        elif avg_score >= 0.5:
            # Liệt kê tất cả các nhịp cần cải thiện
            phases_to_improve = [phase for phase, score in sorted_low_phases]
            result_str += f">> NHẬN XÉT: Em cần chú ý luyện tập thêm các nhịp: {', '.join(phases_to_improve)}."
        else:
            result_str += ">> NHẬN XÉT: Em cần luyện tập thêm nhiều."

        # Sử dụng cửa sổ kết quả tùy chỉnh thay cho messagebox
        ResultsWindow(self, result_str, exercise_name, phase_max_scores, avg_score)

    def open_login_window(self):
        LoginWindow(self)

    def on_login_success(self):
        self.teacher_logged_in = True
        self.login_button.configure(state="disabled")
        self.admin_button.configure(state="normal")
        messagebox.showinfo("Thành công", "Đăng nhập giáo viên thành công!")

    def open_teacher_admin(self):
        TeacherAdminWindow(self)

    def check_for_updates(self):
        """Kiểm tra cập nhật phần mềm trong một luồng riêng."""
        self.update_button.config(state="disabled", text="Đang kiểm tra...")

        def _check_thread():
            try:
                response = requests.get(UPDATE_CHECK_URL, timeout=10)
                response.raise_for_status()
                latest_release = response.json()
                latest_version = latest_release['tag_name'].lstrip('v')
                download_url = latest_release['assets'][0]['browser_download_url']

                if latest_version > CURRENT_VERSION:
                    if messagebox.askyesno("Có bản cập nhật", f"Đã có phiên bản mới: {latest_version}\nPhiên bản hiện tại của bạn là {CURRENT_VERSION}.\nBạn có muốn mở trang tải về không?"):
                        webbrowser.open(download_url)
                else:
                    messagebox.showinfo("Cập nhật", "Bạn đang sử dụng phiên bản mới nhất.")
            except Exception as e:
                messagebox.showerror("Lỗi cập nhật", f"Không thể kiểm tra cập nhật:\n{e}")
            finally:
                self.update_button.config(state="normal", text="Kiểm tra cập nhật")

        threading.Thread(target=_check_thread, daemon=True).start()

    def run_assessment_thread(self, student_video_path, std_data, std_params, frame_mapping, weights, progress_callback):
        """Hàm này chạy trong một luồng riêng để không làm treo giao diện."""
        try:
            # Gọi hàm xử lý chính và truyền callback vào
            # Giả sử hàm này trả về một tuple (final_scores, analyzed_video_path)
            result = assessment.run_assessment_single_view(
                student_video_path,
                std_data,
                std_params,
                frame_mapping,
                weights,
                progress_callback=progress_callback
            )
            self.result_queue.put(result)
        except Exception as e:
            self.result_queue.put(e)

    def check_thread(self, thread, progress_win, selected_name):
        """Kiểm tra xem luồng xử lý đã hoàn thành chưa."""
        if thread.is_alive():
            # Nếu luồng vẫn đang chạy, kiểm tra lại sau 100ms
            self.after(100, self.check_thread, thread, progress_win, selected_name)
        else:
            # Luồng đã kết thúc, đóng cửa sổ tiến trình và lấy kết quả
            progress_win.destroy()
            try:
                result = self.result_queue.get_nowait()
                if isinstance(result, Exception):
                    raise result

                final_scores, analyzed_video_path = result

                # --- Giai đoạn 3: Dọn dẹp và hiển thị kết quả ---
                if analyzed_video_path and os.path.exists(analyzed_video_path):
                    try:
                        os.remove(analyzed_video_path)
                    except OSError as e:
                        print(f"Lỗi khi xóa file video tạm: {e}")

                if final_scores:
                    self.show_final_results(final_scores, selected_name)
                else:
                    messagebox.showerror("Lỗi", "Quá trình đánh giá thất bại. Vui lòng kiểm tra lại video.")

            except queue.Empty:
                messagebox.showerror("Lỗi", "Không nhận được kết quả từ luồng xử lý.")
            except Exception as e:
                messagebox.showerror("Lỗi xử lý", f"Đã xảy ra lỗi trong quá trình phân tích:\n{e}")

class ProgressWindow(tk.Toplevel):
    def __init__(self, parent, title="Đang xử lý..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        self.grab_set()

        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Đang khởi tạo...")

        ttk.Label(self, text="Quá trình phân tích và tính điểm:", font=("", 12)).pack(pady=10)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=350, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=5)
        ttk.Label(self, textvariable=self.status_var).pack(pady=5)

        self.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning("Đang xử lý", "Vui lòng chờ quá trình phân tích hoàn tất.", parent=self))

    def update_progress(self, percentage, status_text):
        self.progress_var.set(percentage)
        self.status_var.set(status_text)
        self.update_idletasks()

class AssessmentWindow(tk.Toplevel):
    """
    Cửa sổ này chỉ dùng để hiển thị video đang được xử lý theo thời gian thực.
    Nó không thực hiện tính toán điểm số.
    """
    def __init__(self, parent, student_video_path, exercise_data):
        super().__init__(parent)
        self.parent = parent
        self.title(f"Xem trước: {exercise_data['name']}")
        self.geometry("1000x800")

        self.grab_set() # Chặn tương tác với cửa sổ chính

        self.student_video_path = student_video_path

        self.video_label = ttk.Label(self)
        self.video_label.pack(pady=10, padx=10, expand=True, fill="both")
        
        self.status_label = ttk.Label(self, text="Đang khởi tạo...", font=("", 12))
        self.status_label.pack(pady=5)

        self.cap = None
        self.model = None
        self.frame_count = 0
        self.is_running = True

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        try:
            model_path = get_resource_path('models/yolov8n-pose.pt')
            self.model = YOLO(model_path)
            self.cap = cv2.VideoCapture(self.student_video_path)
            if not self.cap.isOpened():
                raise IOError("Không thể mở file video của học sinh.")

            # --- Cải tiến: Điều chỉnh kích thước cửa sổ và video ---
            # Lấy kích thước gốc của video
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Đặt kích thước hiển thị tối đa
            MAX_WIDTH = 1280
            MAX_HEIGHT = 720

            # Tính toán tỷ lệ để vừa với kích thước tối đa mà không làm méo ảnh
            scale = min(MAX_WIDTH / original_width, MAX_HEIGHT / original_height)
            # Không phóng to video nếu nó nhỏ hơn kích thước tối đa
            if scale > 1:
                scale = 1

            self.display_size = (int(original_width * scale), int(original_height * scale))

            # Điều chỉnh kích thước cửa sổ cho phù hợp với video + padding
            self.geometry(f"{self.display_size[0] + 40}x{self.display_size[1] + 80}")

            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.delay = int(1000 / video_fps) if video_fps > 0 else 33

        except Exception as e:
            messagebox.showerror("Lỗi khởi tạo", f"Không thể tải model hoặc video:\n{e}", parent=self)
            self.destroy()
            return

        self.process_next_frame()

    def process_next_frame(self):
        if not self.is_running or not self.cap.isOpened():
            self.finish_processing()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_running = False
            self.finish_processing()
            return

        self.frame_count += 1
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        # Thay đổi kích thước khung hình để hiển thị
        resized_frame = cv2.resize(annotated_frame, self.display_size)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)

        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        self.status_label.config(text=f"Đang xử lý frame: {self.frame_count}")

        self.after(self.delay, self.process_next_frame)

    def finish_processing(self):
        if self.cap:
            self.cap.release()
        self.destroy()

    def on_close(self):
        self.is_running = False
        # Vòng lặp `process_next_frame` sẽ tự dừng và gọi `finish_processing`

class LoginWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Đăng nhập Giáo viên")
        self.geometry("300x150")

        frame = ttk.Frame(self, padding="10")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Tên đăng nhập:").pack()
        self.user_entry = ttk.Entry(frame)
        self.user_entry.pack(fill="x", pady=2)

        ttk.Label(frame, text="Mật khẩu:").pack()
        self.pass_entry = ttk.Entry(frame, show="*")
        self.pass_entry.pack(fill="x", pady=2)

        ttk.Button(frame, text="Đăng nhập", command=self.handle_login).pack(pady=10)

    def handle_login(self):
        creds = self.parent.app_config.get('teacher_credentials', {})
        username = self.user_entry.get()
        password = self.pass_entry.get()

        if username == creds.get('username') and password == creds.get('password'):
            self.parent.on_login_success()
            self.destroy()
        else:
            messagebox.showerror("Lỗi", "Tên đăng nhập hoặc mật khẩu không đúng.")

class TeacherAdminWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Quản lý Bài tập")
        self.geometry("900x750")
        self.standard_video_path = tk.StringVar()
        self.weight_entries = {}
        self.sum_labels = {}
        self.editing_exercise_name = None

        # --- Cải tiến: Thêm Canvas và Scrollbar cho form ---
        canvas = tk.Canvas(self) # This canvas will hold the scrollable form
        canvas_scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview) # The scrollbar for the canvas
        self.scrollable_frame = ttk.Frame(canvas)

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill="both")

        # --- Phần hiển thị danh sách bài tập ---
        list_frame = ttk.LabelFrame(main_frame, text="Danh sách bài tập hiện có", padding="10")
        list_frame.pack(fill="x", pady=5, expand=True)

        columns = ('name', 'video')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        self.tree.heading('name', text='Tên bài tập')
        self.tree.heading('video', text='Video mẫu')
        self.tree.column('name', width=300)
        self.tree.column('video', width=400)

        tree_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=tree_scrollbar.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        tree_scrollbar.grid(row=0, column=1, sticky='ns')
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        button_frame = ttk.Frame(list_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky='ew')

        ttk.Button(button_frame, text="Sửa bài tập đã chọn", command=self.load_exercise_for_edit).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Xóa bài tập đã chọn", command=self.delete_exercise).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Thêm bài tập mới (xóa form)", command=self.clear_form).pack(side="left", padx=5)

        # --- Cải tiến: Đặt form vào trong scrollable_frame ---
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=canvas_scrollbar.set)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        form_container = ttk.Frame(main_frame)
        form_container.pack(fill="both", expand=True, pady=10)
        canvas.pack(side="left", fill="both", expand=True)
        canvas_scrollbar.pack(side="right", fill="y")

        # --- Phần form để thêm/sửa bài tập ---
        form_frame = ttk.LabelFrame(self.scrollable_frame, text="Thêm / Sửa chi tiết bài tập", padding="10")
        form_frame.pack(fill="both", pady=10)

        # --- Basic Info ---
        info_frame = ttk.Frame(form_frame)
        info_frame.pack(fill="x", pady=2)
        
        ttk.Label(info_frame, text="Tên bài tập:").pack(anchor="w")
        self.name_entry = ttk.Entry(info_frame)
        self.name_entry.pack(fill="x", expand=True)

        ttk.Label(info_frame, text="Video mẫu:").pack(anchor="w", pady=(5,0))
        video_frame = ttk.Frame(info_frame)
        video_frame.pack(fill="x", expand=True)
        ttk.Entry(video_frame, textvariable=self.standard_video_path, state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Button(video_frame, text="Chọn file...", command=self.select_standard_video).pack(side="left", padx=5)

        # --- Frame Mapping ---
        mapping_frame = ttk.Frame(form_frame)
        mapping_frame.pack(fill="x", pady=5)
        ttk.Label(mapping_frame, text="Phân chia nhịp (Frame Mapping - JSON format):").pack(anchor="w")
        self.frame_mapping_text = tk.Text(mapping_frame, height=6, width=50)
        self.frame_mapping_text.pack(fill="x", expand=True)
        self.frame_mapping_text.bind("<KeyRelease>", self.update_weights_table)

        # --- Phase Weights ---
        weights_frame_container = ttk.Frame(form_frame)
        weights_frame_container.pack(fill="x", pady=5)
        ttk.Label(weights_frame_container, text="Trọng số các góc cho từng nhịp:").pack(anchor="w")
        self.weights_frame = ttk.Frame(weights_frame_container, borderwidth=1, relief="solid")
        self.weights_frame.pack(fill="x", expand=True, pady=5)

        # --- Nút Lưu ---
        # Sửa lỗi: Di chuyển nút Lưu ra ngoài vùng cuộn để tránh lỗi
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill='x', pady=5, side='bottom')
        self.save_button = ttk.Button(save_frame, text="Lưu bài tập mới", command=self.save_exercise)
        self.save_button.pack()

        self.populate_treeview()

    def populate_treeview(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for exercise in self.parent.app_config.get('exercises', []):
            self.tree.insert('', 'end', values=(exercise['name'], exercise['standard_video']))

    def load_exercise_for_edit(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn một bài tập để sửa.", parent=self)
            return

        selected_values = self.tree.item(selected_item[0], 'values')
        exercise_name = selected_values[0]

        exercise_data = next((ex for ex in self.parent.app_config['exercises'] if ex['name'] == exercise_name), None)
        if not exercise_data:
            messagebox.showerror("Lỗi", "Không tìm thấy dữ liệu cho bài tập này.", parent=self)
            return

        self.clear_form()
        self.editing_exercise_name = exercise_name
        self.save_button.config(text=f"Lưu thay đổi cho '{exercise_name}'")

        self.name_entry.delete(0, 'end')
        self.name_entry.insert(0, exercise_data['name'])
        self.standard_video_path.set(exercise_data['standard_video'])
        self.frame_mapping_text.delete('1.0', 'end')
        self.frame_mapping_text.insert('1.0', json.dumps(exercise_data['frame_mapping'], indent=4, ensure_ascii=False))
        
        self.update_weights_table(None)
        
        if 'phase_weights' in exercise_data:
            for phase, weights in exercise_data['phase_weights'].items():
                if phase in self.weight_entries:
                    for angle, weight in weights.items():
                        if angle in self.weight_entries[phase]:
                            self.weight_entries[phase][angle].delete(0, 'end')
                            self.weight_entries[phase][angle].insert(0, str(weight))
        
        self.update_all_sum_labels()

    def delete_exercise(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn một bài tập để xóa.", parent=self)
            return

        selected_values = self.tree.item(selected_item[0], 'values')
        exercise_name = selected_values[0]

        if messagebox.askyesno("Xác nhận xóa", f"Bạn có chắc chắn muốn xóa bài tập '{exercise_name}' không?", parent=self):
            initial_len = len(self.parent.app_config['exercises'])
            self.parent.app_config['exercises'] = [ex for ex in self.parent.app_config['exercises'] if ex['name'] != exercise_name]
            
            if len(self.parent.app_config['exercises']) < initial_len:
                self.parent.save_config()
                self.parent.populate_exercises()
                self.populate_treeview()
                self.clear_form()
                messagebox.showinfo("Thành công", f"Đã xóa bài tập '{exercise_name}'.", parent=self)
            else:
                messagebox.showerror("Lỗi", "Không thể xóa bài tập.", parent=self)

    def clear_form(self):
        self.editing_exercise_name = None
        self.save_button.config(text="Lưu bài tập mới")
        self.name_entry.delete(0, 'end')
        self.standard_video_path.set("")
        self.frame_mapping_text.delete('1.0', 'end')
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
        self.weight_entries = {}
        self.sum_labels = {}

    def save_exercise(self):
        name = self.name_entry.get().strip()
        video = self.standard_video_path.get().strip()
        frame_mapping_str = self.frame_mapping_text.get("1.0", "end-1c")

        if not name or not video:
            messagebox.showerror("Lỗi", "Tên bài tập và Video mẫu không được để trống.", parent=self)
            return

        try:
            frame_mapping = json.loads(frame_mapping_str) if frame_mapping_str else {}
            if not isinstance(frame_mapping, dict):
                raise ValueError("Frame mapping phải là một đối tượng JSON.")
        except (json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Lỗi định dạng", f"Dữ liệu Frame Mapping không hợp lệ:\n{e}", parent=self)
            return

        phase_weights = {}
        for phase, entries in self.weight_entries.items():
            phase_weights[phase] = {}
            current_sum = 0
            for angle, entry in entries.items():
                try:
                    weight = float(entry.get()) if entry.get() else 0.0
                    if not (0 <= weight <= 1):
                        raise ValueError()
                    phase_weights[phase][angle] = weight
                    current_sum += weight
                except ValueError:
                    messagebox.showerror("Lỗi trọng số", f"Trọng số cho '{angle}' trong nhịp '{phase}' không hợp lệ. Phải là số từ 0 đến 1.", parent=self)
                    return
            
            if not np.isclose(current_sum, 1.0) and current_sum > 0:
                messagebox.showwarning("Cảnh báo trọng số", f"Tổng trọng số cho nhịp '{phase}' ({current_sum:.2f}) không bằng 1.", parent=self)

        new_exercise_data = {
            "name": name,
            "standard_video": video,
            "cache_file": f"assets/cache_{name.lower().replace(' ', '_')}.json",
            "frame_mapping": frame_mapping,
            "phase_weights": phase_weights
        }

        if self.editing_exercise_name:
            if self.editing_exercise_name != name and any(ex['name'] == name for ex in self.parent.app_config['exercises']):
                 messagebox.showerror("Lỗi", f"Tên bài tập '{name}' đã tồn tại.", parent=self)
                 return
            
            for i, ex in enumerate(self.parent.app_config['exercises']):
                if ex['name'] == self.editing_exercise_name:
                    self.parent.app_config['exercises'][i] = new_exercise_data
                    break
            messagebox.showinfo("Thành công", f"Đã cập nhật bài tập '{name}' thành công!", parent=self)
        else:
            if any(ex['name'] == name for ex in self.parent.app_config['exercises']):
                messagebox.showerror("Lỗi", f"Tên bài tập '{name}' đã tồn tại.", parent=self)
                return
            self.parent.app_config['exercises'].append(new_exercise_data)
            messagebox.showinfo("Thành công", f"Đã thêm bài tập '{name}' thành công!", parent=self)

        self.parent.save_config()
        self.parent.populate_exercises()
        self.populate_treeview()
        self.clear_form()

    def select_standard_video(self):
        path = filedialog.askopenfilename(title="Chọn video mẫu", filetypes=[("Video files", "*.mp4 *.avi")])
        if path:
            # Calculate relative path from project root (not from current working directory)
            # In bundled exe: project root = directory containing .exe
            # In development: project root = current directory
            if getattr(sys, 'frozen', False):
                # Running from exe - use exe's directory as base
                project_root = os.path.dirname(sys.executable)
            else:
                # Development mode - use current directory
                project_root = os.path.abspath(".")
            
            # Create relative path from project root
            try:
                relative_path = os.path.relpath(path, project_root)
                # Ensure forward slashes for cross-platform compatibility
                relative_path = relative_path.replace("\\", "/")
            except ValueError:
                # If files are on different drives, just use the filename with assets/
                relative_path = f"assets/{os.path.basename(path)}"
            
            self.standard_video_path.set(relative_path)
            # --- Cải tiến: Tự động phân tích và điền frame mapping ---
            self.auto_fill_frame_mapping(path)

    def auto_fill_frame_mapping(self, video_path):
        """Phân tích video để tự động điền frame mapping và weights."""
        progress_win = ProgressWindow(self, "Đang phân tích video mẫu...")
        result_queue = queue.Queue()

        def _analysis_thread():
            try:
                # 1. Tạo đường dẫn cache tạm thời để tránh ghi đè cache hiện có
                video_filename = os.path.basename(video_path)
                temp_cache_name = f"assets/cache_temp_{video_filename}.json"

                # 2. Trích xuất keypoints (sẽ được cache lại)
                progress_win.update_progress(5, "Trích xuất keypoints từ video mẫu...")
                std_data = assessment.load_or_extract_standard(video_path, temp_cache_name)
                if not std_data:
                    raise ValueError("Không thể trích xuất keypoints từ video.")

                # 3. Tạo frame mapping từ dữ liệu keypoints đã trích xuất
                progress_win.update_progress(70, "Phân tích chuyển động để chia nhịp...")
                mapping = assessment.generate_frame_mapping_from_video(
                    video_path, lambda p, s: self.after(0, progress_win.update_progress, p, s))

                # 4. Gợi ý trọng số dựa trên dữ liệu và mapping
                progress_win.update_progress(90, "Gợi ý trọng số dựa trên chuyển động...")
                weights = assessment.suggest_phase_weights(std_data, mapping)

                # Xóa file cache tạm
                if os.path.exists(temp_cache_name):
                    os.remove(temp_cache_name)

                result_queue.put({'mapping': mapping, 'weights': weights})
            except Exception as e:
                result_queue.put(e)

        thread = threading.Thread(target=_analysis_thread, daemon=True)
        thread.start()
        self.check_mapping_thread(thread, progress_win, result_queue)

    def check_mapping_thread(self, thread, progress_win, result_queue):
        if thread.is_alive():
            self.after(100, self.check_mapping_thread, thread, progress_win, result_queue)
        else:
            progress_win.destroy()
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    raise result
                
                # Cập nhật giao diện với mapping và weights được gợi ý
                self.frame_mapping_text.delete('1.0', 'end')
                self.frame_mapping_text.insert('1.0', json.dumps(result['mapping'], indent=4, ensure_ascii=False))
                self.update_weights_table(None) # Cập nhật bảng trọng số theo mapping mới

                # Điền các trọng số được gợi ý vào bảng
                if 'weights' in result:
                    for phase, weights in result['weights'].items():
                        if phase in self.weight_entries:
                            for angle, weight in weights.items():
                                if angle in self.weight_entries[phase]:
                                    self.weight_entries[phase][angle].delete(0, 'end')
                                    self.weight_entries[phase][angle].insert(0, str(weight))
                self.update_all_sum_labels()

            except queue.Empty:
                messagebox.showerror("Lỗi", "Không nhận được kết quả từ luồng phân tích video mẫu.", parent=self)
            except Exception as e:
                messagebox.showerror("Lỗi phân tích", f"Đã xảy ra lỗi khi phân tích video mẫu:\n{e}", parent=self)

    def update_weights_table(self, event=None):
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
        self.weight_entries = {}
        self.sum_labels = {}
        
        try:
            frame_mapping = json.loads(self.frame_mapping_text.get("1.0", "end-1c"))
            phases = list(frame_mapping.keys())
        except json.JSONDecodeError:
            phases = []

        angles = [
            "L_SHOULDER_ANGLE", "R_SHOULDER_ANGLE", "L_ELBOW_ANGLE", "R_ELBOW_ANGLE",
            "L_HIP_ANGLE", "R_HIP_ANGLE", "L_KNEE_ANGLE", "R_KNEE_ANGLE", "WRIST_DISTANCE", "ANKLE_DISTANCE"
        ]
        self.angle_mapping = {
            "Vai trái": "L_SHOULDER_ANGLE", "Vai phải": "R_SHOULDER_ANGLE",
            "Khuỷu trái": "L_ELBOW_ANGLE", "Khuỷu phải": "R_ELBOW_ANGLE",
            "Hông trái": "L_HIP_ANGLE", "Hông phải": "R_HIP_ANGLE",
            "Đầu gối trái": "L_KNEE_ANGLE", "Đầu gối phải": "R_KNEE_ANGLE",
            "KC 2 bàn tay": "WRIST_DISTANCE", "KC 2 cổ chân": "ANKLE_DISTANCE"
        }

        for i, phase in enumerate(phases):
            phase_frame = ttk.LabelFrame(self.weights_frame, text=phase, padding=5)
            phase_frame.pack(side="top", fill="x", padx=5, pady=5)
            self.weight_entries[phase] = {}
            
            for j, angle in enumerate(angles):
                display_name = next((k for k, v in self.angle_mapping.items() if v == angle), angle)
                ttk.Label(phase_frame, text=display_name).grid(row=j, column=0, sticky="w", padx=5)
                entry = ttk.Entry(phase_frame, width=8)
                entry.grid(row=j, column=1, padx=5)
                entry.bind("<KeyRelease>", lambda e, p=phase: self.update_sum_label(p))
                self.weight_entries[phase][angle] = entry # Vẫn dùng key gốc để lưu dữ liệu
            
            sum_frame = ttk.Frame(phase_frame)
            sum_frame.grid(row=len(angles), column=0, columnspan=2, pady=5)
            ttk.Label(sum_frame, text="Tổng:").pack(side="left")
            sum_label = ttk.Label(sum_frame, text="0.0")
            sum_label.pack(side="left")
            self.sum_labels[phase] = sum_label

    def update_sum_label(self, phase):
        if phase not in self.weight_entries: return
        current_sum = 0
        for angle, entry in self.weight_entries[phase].items():
            try:
                current_sum += float(entry.get())
            except (ValueError, TypeError):
                pass
        
        sum_label = self.sum_labels.get(phase)
        if sum_label:
            sum_label.config(text=f"{current_sum:.2f}")
            if not np.isclose(current_sum, 1.0) and current_sum > 0:
                sum_label.config(foreground="red")
            else:
                sum_label.config(foreground="")

    def update_all_sum_labels(self):
        for phase in self.sum_labels.keys():
            self.update_sum_label(phase)

class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, result_text, exercise_name, scores, avg_score):
        super().__init__(parent)
        self.title("Kết quả cuối cùng")
        self.geometry("450x550")

        # Lưu trữ dữ liệu kết quả
        self.result_text = result_text
        self.exercise_name = exercise_name
        self.scores = scores
        self.avg_score = avg_score

        # Thêm một frame để có padding
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(expand=True, fill="both")

        # --- Vùng hiển thị kết quả ---
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(expand=True, fill="both")

        # Thêm thanh cuộn
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        # Thêm vùng văn bản
        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        text_widget.pack(expand=True, fill="both")
        
        text_widget.insert("1.0", result_text)
        text_widget.config(state="disabled") # Đặt ở chế độ chỉ đọc

        scrollbar.config(command=text_widget.yview)

        # --- Vùng gửi kết quả ---
        send_frame = ttk.LabelFrame(main_frame, text="Gửi kết quả cho Giáo viên", padding="10")
        send_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(send_frame, text="Tên của bạn:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.student_name_entry = ttk.Entry(send_frame, width=30)
        self.student_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.send_button = ttk.Button(send_frame, text="Gửi kết quả", command=self.send_results)
        self.send_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(main_frame, text="Đóng", command=self.destroy).pack(pady=(10, 0))

    def send_results(self):
        student_name = self.student_name_entry.get().strip()
        if not student_name:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập tên của bạn.", parent=self)
            return

        self.send_button.config(state="disabled", text="Đang gửi...")

        payload = {
            "student_name": student_name,
            "exercise_name": self.exercise_name,
            "average_score": self.avg_score,
            "detailed_scores": self.scores,
            "full_text_result": self.result_text
        }

        def _send_thread():
            try:
                # THAY ĐỔI: Đây là nơi thực hiện gửi dữ liệu lên server
                # response = requests.post(RESULTS_API_ENDPOINT, json=payload, timeout=15)
                # response.raise_for_status() # Ném lỗi nếu request không thành công (status code không phải 2xx)
                
                # Giả lập thành công để test
                print("Gửi dữ liệu:", json.dumps(payload, indent=2, ensure_ascii=False))
                messagebox.showinfo("Thành công", "Đã gửi kết quả thành công!", parent=self)

            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể gửi kết quả:\n{e}", parent=self)
            finally:
                self.send_button.config(state="normal", text="Gửi kết quả")

        threading.Thread(target=_send_thread, daemon=True).start()

import cv2
import json
import numpy as np
from scipy.spatial.distance import euclidean
import os
try:
    from scipy.signal import find_peaks
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

from ultralytics import YOLO
from .utils import calculate_angle, get_midpoint, get_resource_path
from .reference_processor import ReferencePoseProcessor

MODEL = YOLO(get_resource_path('models/yolov8s-pose.pt'))


def extract_keypoints_from_video(video_path, output_path=None):
    """
    Trích xuất keypoints từ video và lưu vào list.
    Tùy chọn, lưu một video mới với các keypoint được vẽ lên.
    """
    print(f"-> Bắt đầu trích xuất keypoints từ: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    # Lấy thông số video để tạo VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo đối tượng VideoWriter nếu có đường dẫn output
    video_writer = None
    if output_path:
        print(f"   ... Sẽ lưu video phân tích vào: {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_keypoints = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = MODEL(frame, verbose=False)

        # Vẽ các điểm keypoint và khung xương lên frame
        annotated_frame = results[0].plot()

        if results[0].keypoints and results[0].keypoints.data.shape[0] > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            all_keypoints.append(keypoints.tolist())
        else:
            all_keypoints.append(None)

        # Ghi frame đã vẽ vào video output
        if video_writer:
            video_writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   ... đã xử lý {frame_count} frames.")

    cap.release()
    if video_writer:
        video_writer.release()
    print(f"-> Trích xuất hoàn tất. Tổng số frames: {len(all_keypoints)}")
    return all_keypoints

def load_or_extract_standard(video_path, cache_path):
    """
    Tải dữ liệu keypoints chuẩn từ file cache nếu có, nếu không thì trích xuất từ video.
    """
    # Xử lý đường dẫn để hoạt động cả trong dev và PyInstaller
    full_video_path = get_resource_path(video_path)
    full_cache_path = get_resource_path(cache_path)
    
    # Tạo thư mục cache nếu cần
    cache_dir = os.path.dirname(full_cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(full_cache_path):
        print(f"Đang tải dữ liệu chuẩn từ cache: {full_cache_path}")
        with open(full_cache_path, 'r') as f:
            # Need to convert list of lists back to numpy for processing if needed downstream, 
            # but standard_data is mainly used for coords. 
            # ReferenceProcessor returns list of numpy arrays or list of lists? 
            # It returns ideal_rep_kps which is numpy array. json.dump converts to list.
            data = json.load(f)
            return data
    else:
        print(f"Không tìm thấy cache. Bắt đầu xử lý nâng cao video chuẩn: {full_video_path}")
        processor = ReferencePoseProcessor()
        # Use full sequence (use_ideal_rep=False) to match the frame mapping indices of the full video.
        # This prevents mismatch where mapping expects frame 400 but ideal rep only has 100 frames.
        ideal_rep = processor.process_video(full_video_path, use_ideal_rep=False)
        
        if ideal_rep is not None:
            print(f"Lưu dữ liệu chuẩn (Full Normalized Sequence) vào cache: {full_cache_path}")
            # Convert numpy to list for JSON serialization
            ideal_rep_list = ideal_rep.tolist() if isinstance(ideal_rep, np.ndarray) else ideal_rep
            with open(full_cache_path, 'w') as f:
                json.dump(ideal_rep_list, f)
            return ideal_rep_list
        return None

def calculate_params_for_frame(keypoints):
    """
    Tính toán các thông số góc và khoảng cách cho một frame.
    """
    if keypoints is None or len(keypoints) < 17:
        return None

    # Định nghĩa các chỉ số keypoint của COCO
    R_SHOULDER, L_SHOULDER = 6, 5
    R_ELBOW, L_ELBOW = 8, 7
    R_WRIST, L_WRIST = 10, 9
    R_HIP, L_HIP = 12, 11
    R_KNEE, L_KNEE = 14, 13
    R_ANKLE, L_ANKLE = 16, 15

    # Lấy tọa độ
    r_shoulder_pt = keypoints[R_SHOULDER][:2]
    l_shoulder_pt = keypoints[L_SHOULDER][:2]
    r_elbow_pt = keypoints[R_ELBOW][:2]
    l_elbow_pt = keypoints[L_ELBOW][:2]
    r_wrist_pt = keypoints[R_WRIST][:2]
    l_wrist_pt = keypoints[L_WRIST][:2]
    r_hip_pt = keypoints[R_HIP][:2]
    l_hip_pt = keypoints[L_HIP][:2]
    r_knee_pt = keypoints[R_KNEE][:2]
    l_knee_pt = keypoints[L_KNEE][:2]

    # Tính toán các góc
    r_arm_angle = calculate_angle(r_hip_pt, r_shoulder_pt, r_elbow_pt)
    l_arm_angle = calculate_angle(l_hip_pt, l_shoulder_pt, l_elbow_pt)
    r_elbow_angle = calculate_angle(r_shoulder_pt, r_elbow_pt, r_wrist_pt)
    l_elbow_angle = calculate_angle(l_shoulder_pt, l_elbow_pt, l_wrist_pt)
    
    # Tính góc vai (giữa cánh tay và thân)
    mid_hip = get_midpoint(r_hip_pt, l_hip_pt)
    r_shoulder_angle = calculate_angle(mid_hip, r_shoulder_pt, r_elbow_pt)
    l_shoulder_angle = calculate_angle(mid_hip, l_shoulder_pt, l_elbow_pt)

    # Tính góc hông (giữa thân và đùi)
    r_hip_angle = calculate_angle(r_shoulder_pt, r_hip_pt, r_knee_pt)
    l_hip_angle = calculate_angle(l_shoulder_pt, l_hip_pt, l_knee_pt)

    # Tính góc đầu gối
    r_ankle_pt = keypoints[R_ANKLE][:2]
    l_ankle_pt = keypoints[L_ANKLE][:2]
    r_knee_angle = calculate_angle(r_hip_pt, r_knee_pt, r_ankle_pt)
    l_knee_angle = calculate_angle(l_hip_pt, l_knee_pt, l_ankle_pt)

    # Tính khoảng cách (chuẩn hóa bằng khoảng cách giữa 2 vai)
    shoulder_dist = np.linalg.norm(np.array(r_shoulder_pt) - np.array(l_shoulder_pt))
    if shoulder_dist == 0: shoulder_dist = 1 # Tránh chia cho 0
    
    wrists_dist = np.linalg.norm(np.array(r_wrist_pt) - np.array(l_wrist_pt)) / shoulder_dist
    hips_dist = np.linalg.norm(np.array(r_hip_pt) - np.array(l_hip_pt)) / shoulder_dist
    ankle_dist = np.linalg.norm(np.array(r_ankle_pt) - np.array(l_ankle_pt)) / shoulder_dist

    return {
        'R_ARM_ANGLE': r_arm_angle, 
        'L_ARM_ANGLE': l_arm_angle,
        'R_ELBOW_ANGLE': r_elbow_angle,
        'L_ELBOW_ANGLE': l_elbow_angle,
        'R_SHOULDER_ANGLE': r_shoulder_angle,
        'L_SHOULDER_ANGLE': l_shoulder_angle,
        'R_HIP_ANGLE': r_hip_angle,
        'L_HIP_ANGLE': l_hip_angle,
        'R_KNEE_ANGLE': r_knee_angle,
        'L_KNEE_ANGLE': l_knee_angle,
        'HIP_DISTANCE': hips_dist, 
        'WRIST_DISTANCE': wrists_dist,
        'ANKLE_DISTANCE': ankle_dist,
    }

def get_average_params(params):
    """
    Tính toán các thông số trung bình từ danh sách các thông số đã trích xuất.
    Xử lý các trường hợp danh sách rỗng để tránh lỗi.
    """
    return {
        'AVG_R_ARM_ANGLE': np.mean(params['r_arm_angles']) if params['r_arm_angles'] else 180.0,
        'AVG_L_ARM_ANGLE': np.mean(params['l_arm_angles']) if params['l_arm_angles'] else 180.0,
        'AVG_R_ELBOW_ANGLE': np.mean(params['r_elbow_angles']) if params['r_elbow_angles'] else 175.0,
        'AVG_L_ELBOW_ANGLE': np.mean(params['l_elbow_angles']) if params['l_elbow_angles'] else 175.0,
        'AVG_R_SHOULDER_ANGLE': np.mean(params['r_shoulder_angles']) if params['r_shoulder_angles'] else 90.0,
        'AVG_L_SHOULDER_ANGLE': np.mean(params['l_shoulder_angles']) if params['l_shoulder_angles'] else 90.0,
        'AVG_R_HIP_ANGLE': np.mean(params['r_hip_angles']) if params['r_hip_angles'] else 178.0,
        'AVG_L_HIP_ANGLE': np.mean(params['l_hip_angles']) if params['l_hip_angles'] else 178.0,
        'AVG_R_KNEE_ANGLE': np.mean(params['r_knee_angles']) if params['r_knee_angles'] else 178.0,
        'AVG_L_KNEE_ANGLE': np.mean(params['l_knee_angles']) if params['l_knee_angles'] else 178.0,
        'AVG_HIP_DISTANCE': np.mean(params['hips_dist']) if params['hips_dist'] else 0.4,
        'AVG_WRIST_DISTANCE': np.mean(params['wrists_dist']) if params['wrists_dist'] else 0.05
    }

def align_phases_dtw(standard_kps, student_kps, standard_frame_mapping):
    """
    Sử dụng Dynamic Time Warping (DTW) để tìm ra frame mapping động cho video của học sinh.
    Đây là phương pháp "co giãn" thời gian để căn chỉnh động tác.
    :param standard_kps: List các keypoints của video mẫu.
    :param student_kps: List các keypoints của video học sinh.
    :param standard_frame_mapping: Frame mapping gốc của video mẫu.
    :return: Frame mapping mới, đã được điều chỉnh cho video của học sinh.
    """
    if fastdtw is None:
        print("Lỗi: Vui lòng cài đặt thư viện 'fastdtw' (pip install fastdtw) để sử dụng tính năng căn chỉnh động.")
        return standard_frame_mapping # Quay về sử dụng mapping tĩnh nếu thiếu thư viện

    print("-> Bắt đầu căn chỉnh động tác bằng Dynamic Time Warping (DTW)...")

    # Chuyển đổi keypoints thành dạng numpy array phẳng để tính khoảng cách    # Bỏ qua tọa độ z nếu có, chỉ dùng (x, y) và xử lý các frame có thể bị thiếu (None)
    std_kps_flat = np.array([np.array(frame)[:, :2].flatten() for frame in standard_kps if frame is not None])
    student_kps_flat = np.array([np.array(frame)[:, :2].flatten() for frame in student_kps if frame is not None])

    if len(std_kps_flat) == 0 or len(student_kps_flat) == 0:
        print("Cảnh báo: Không có đủ dữ liệu keypoint để thực hiện DTW. Sử dụng mapping tĩnh.")
        return standard_frame_mapping

    # Tìm đường đi tối ưu giữa toàn bộ video mẫu và video học sinh
    distance, path = fastdtw(std_kps_flat, student_kps_flat, dist=euclidean)

    # path là một list các tuple (index_std, index_student)
    # Tạo một dictionary để map từ frame mẫu sang frame học sinh
    std_to_student_map = dict(path)

    dynamic_frame_mapping = {}
    last_student_frame = 0

    # Sắp xếp các nhịp theo frame bắt đầu để xử lý tuần tự
    sorted_phases = sorted(standard_frame_mapping.items(), key=lambda item: item[1][0])

    for i, (phase_name, (start_frame_std, end_frame_std)) in enumerate(sorted_phases):
        # Tìm frame kết thúc tương ứng trong video học sinh
        # Lấy frame cuối cùng của nhịp trong video mẫu
        target_std_frame = end_frame_std - 1

        # Tìm frame mẫu gần nhất có trong map (đề phòng trường hợp path không đầy đủ)
        while target_std_frame not in std_to_student_map and target_std_frame > start_frame_std:
            target_std_frame -= 1
        student_end_frame = std_to_student_map.get(target_std_frame)

        if student_end_frame is None:
            # Nếu không tìm thấy, ước lượng dựa trên tỉ lệ video
            print(f"Cảnh báo: Không thể map frame cuối của nhịp '{phase_name}'. Sẽ ước lượng.")
            ratio = len(student_kps) / len(standard_kps) if len(standard_kps) > 0 else 1
            student_end_frame = int(end_frame_std * ratio)

        student_end_frame = max(student_end_frame, last_student_frame + 1) # Đảm bảo frame kết thúc sau frame bắt đầu
        dynamic_frame_mapping[phase_name] = [last_student_frame, student_end_frame]
        last_student_frame = student_end_frame

    print("-> Căn chỉnh DTW hoàn tất. Frame mapping động:", dynamic_frame_mapping)
    return dynamic_frame_mapping

def calculate_standard_params_per_phase(standard_data, frame_mapping):
    """
    Tính toán các thông số trung bình CHUẨN cho TỪNG NHỊP của video mẫu.
    Đây là một cải tiến quan trọng để tăng độ chính xác.
    """
    print("-> Đang tính toán thông số chuẩn cho từng nhịp...")
    all_phase_params = {}
    for phase_name, (start_frame, end_frame) in frame_mapping.items():
        end_frame = min(end_frame, len(standard_data))
        start_frame = min(start_frame, end_frame)
        
        phase_keypoints = standard_data[start_frame:end_frame]
        
        phase_params_lists = {
            'r_arm_angles': [], 'l_arm_angles': [],
            'r_elbow_angles': [], 'l_elbow_angles': [],
            'r_shoulder_angles': [], 'l_shoulder_angles': [],
            'r_hip_angles': [], 'l_hip_angles': [],
            'r_knee_angles': [], 'l_knee_angles': [],
            'hips_dist': [], 'wrists_dist': [], 'ankle_dist': []
        }

        for frame_kps in phase_keypoints:
            params = calculate_params_for_frame(frame_kps)
            if params:
                phase_params_lists['r_arm_angles'].append(params['R_ARM_ANGLE'])
                phase_params_lists['l_arm_angles'].append(params['L_ARM_ANGLE'])
                phase_params_lists['r_elbow_angles'].append(params['R_ELBOW_ANGLE'])
                phase_params_lists['l_elbow_angles'].append(params['L_ELBOW_ANGLE'])
                phase_params_lists['r_shoulder_angles'].append(params['R_SHOULDER_ANGLE'])
                phase_params_lists['l_shoulder_angles'].append(params['L_SHOULDER_ANGLE'])
                phase_params_lists['r_hip_angles'].append(params['R_HIP_ANGLE'])
                phase_params_lists['l_hip_angles'].append(params['L_HIP_ANGLE'])
                phase_params_lists['r_knee_angles'].append(params['R_KNEE_ANGLE'])
                phase_params_lists['l_knee_angles'].append(params['L_KNEE_ANGLE'])
                phase_params_lists['hips_dist'].append(params['HIP_DISTANCE'])
                phase_params_lists['wrists_dist'].append(params['WRIST_DISTANCE'])
                phase_params_lists['ankle_dist'].append(params['ANKLE_DISTANCE'])

        all_phase_params[phase_name] = get_average_params(phase_params_lists)
        
    print("-> Tính toán thông số chuẩn hoàn tất.")
    return all_phase_params

def calculate_phase_score(std_params, student_params, weights, difficulty_threshold=25.0):
    """
    Tính điểm cho một nhịp dựa trên sự khác biệt giữa thông số chuẩn và của học sinh.
    Sử dụng threshold-based scoring:
    - Error <= threshold: 100% score
    - Error between threshold and 2×threshold: Linear decay from 100% to 0%
    - Error > 2×threshold: 0% score
    
    :param difficulty_threshold: Ngưỡng sai số cho phép (từ scoring_difficulty trong config)
    """
    total_score = 0
    total_weight = 0
    
    metric_to_param_key = {
        'R_ARM_ANGLE': 'AVG_R_ARM_ANGLE', 'L_ARM_ANGLE': 'AVG_L_ARM_ANGLE',
        'R_ELBOW_ANGLE': 'AVG_R_ELBOW_ANGLE', 'L_ELBOW_ANGLE': 'AVG_L_ELBOW_ANGLE',
        'R_SHOULDER_ANGLE': 'AVG_R_SHOULDER_ANGLE', 'L_SHOULDER_ANGLE': 'AVG_L_SHOULDER_ANGLE',
        'R_HIP_ANGLE': 'AVG_R_HIP_ANGLE', 'L_HIP_ANGLE': 'AVG_L_HIP_ANGLE',
        'R_KNEE_ANGLE': 'AVG_R_KNEE_ANGLE', 'L_KNEE_ANGLE': 'AVG_L_KNEE_ANGLE',
        'WRIST_DISTANCE': 'AVG_WRIST_DISTANCE',
        'ANKLE_DISTANCE': 'AVG_ANKLE_DISTANCE'
    }
    
    for metric, weight in weights.items():
        if weight > 0:
            param_key = metric_to_param_key.get(metric)
            if param_key and param_key in std_params and param_key in student_params:
                std_val = std_params[param_key]
                student_val = student_params[param_key]
                diff = abs(std_val - student_val)
                
                # Apply threshold-based scoring
                if 'ANGLE' in param_key:
                    # For angles, use threshold directly (in degrees)
                    if diff <= difficulty_threshold:
                        score = 1.0
                    elif diff <= difficulty_threshold * 2:
                        # Linear decay from 1.0 to 0.0 between threshold and 2×threshold
                        score = 1.0 - ((diff - difficulty_threshold) / difficulty_threshold)
                    else:
                        score = 0.0
                else:
                    # For distances, scale threshold to normalized distance range [0, 2]
                    # Scale: threshold degrees -> proportional distance units
                    distance_threshold = difficulty_threshold / 180.0 * 2.0
                    if diff <= distance_threshold:
                        score = 1.0
                    elif diff <= distance_threshold * 2:
                        score = 1.0 - ((diff - distance_threshold) / distance_threshold)
                    else:
                        score = 0.0
                
                total_score += score * weight
                total_weight += weight

    if total_weight == 0:
        return 1.0
        
    phase_score = total_score / total_weight
    print(f"Điểm số cho nhịp: {phase_score:.2f} (threshold: {difficulty_threshold}°)")
    return phase_score

def run_assessment_single_view(student_video_path, std_data, std_params_per_phase, standard_frame_mapping, phase_weights, difficulty_threshold=25.0, progress_callback=None):
    """
    Chạy đánh giá cho một video của học sinh.
    Cải tiến: Sử dụng DTW để có frame_mapping động.
    Cải tiến: Lưu lại video đã được phân tích.
    """
    print(f"Bắt đầu đánh giá video: {student_video_path}")

    # Tạo đường dẫn cho video output
    path_parts = os.path.splitext(student_video_path)
    output_video_path = f"{path_parts[0]}_analyzed.mp4"

    if progress_callback:
        progress_callback(5, "Trích xuất keypoints từ video...")

    # 1. Trích xuất keypoints từ video học sinh và lưu video đã phân tích
    student_data = extract_keypoints_from_video(student_video_path, output_path=output_video_path)
    if not student_data:
        print("Không thể trích xuất keypoints từ video của học sinh.")
        return None, None

    # Ghi chú: Các bước tiếp theo (DTW, tính toán, chấm điểm) sẽ diễn ra ở đây.
    # Bạn có thể thêm các lời gọi `progress_callback` ở các giai đoạn khác nhau
    # để cập nhật thanh tiến trình một cách chi tiết hơn.
    # Ví dụ:
    if progress_callback:
        progress_callback(75, "Đồng bộ hóa động tác...")
    # ... (mã cho DTW và các bước tính toán khác của bạn) ...

    if progress_callback:
        progress_callback(95, "Hoàn tất tính điểm...")
    
    # ... (phần còn lại của hàm của bạn để trả về final_scores và output_video_path)

    # 2. Preprocess & Normalize Student Data for comparison
    # We must match the processing done on the reference (Interpolation -> Smoothing -> Normalization)
    print("-> Preprocessing student data for alignment...")
    processor = ReferencePoseProcessor() # Instantiate to use helper methods
    student_data_clean = processor.preprocess_keypoints(student_data)
    student_data_norm = processor.normalize_sequence(student_data_clean)

    # 3. Căn chỉnh động tác và lấy frame mapping động bằng DTW
    # Use NORMALIZED data for both standard and student
    dynamic_frame_mapping = align_phases_dtw(std_data, student_data_norm, standard_frame_mapping)

    # 4. Tính toán các thông số của học sinh cho từng nhịp đã được căn chỉnh
    # Use NORMALIZED data for parameter calculation too (Distance metrics rely on normalized coords implicitly or explicitly)
    student_params_per_phase = calculate_standard_params_per_phase(student_data_norm, dynamic_frame_mapping)

    final_scores = {}
    for phase in standard_frame_mapping.keys(): # Dùng key từ mapping gốc để đảm bảo đủ các nhịp
        if phase not in std_params_per_phase or phase not in student_params_per_phase:
            print(f"Cảnh báo: Bỏ qua nhịp '{phase}' do thiếu dữ liệu chuẩn hoặc dữ liệu của học sinh.")
            continue

        std_params = std_params_per_phase[phase]
        student_params = student_params_per_phase[phase]
        weights = phase_weights.get(phase, {})

        phase_score = calculate_phase_score(std_params, student_params, weights, difficulty_threshold)
        final_scores[phase] = phase_score

    print("Điểm số cuối cùng:", final_scores)
    return final_scores, output_video_path

def calculate_scores_from_data(student_data, std_data, std_params_per_phase, standard_frame_mapping, phase_weights, difficulty_threshold=25.0):
    """
    Tính toán điểm số từ dữ liệu keypoints đã được trích xuất trước.
    Đây là hàm tối ưu hóa để tránh xử lý lại video.
    """
    print("Bắt đầu tính điểm từ dữ liệu đã thu thập...")

    # 1. Preprocess & Normalize Student Data
    print("-> Preprocessing student data for alignment...")
    processor = ReferencePoseProcessor()
    student_data_clean = processor.preprocess_keypoints(student_data)
    student_data_norm = processor.normalize_sequence(student_data_clean)

    # 2. Căn chỉnh động tác và lấy frame mapping động bằng DTW
    dynamic_frame_mapping = align_phases_dtw(std_data, student_data_norm, standard_frame_mapping)

    # 3. Tính toán các thông số của học sinh cho từng nhịp đã được căn chỉnh
    student_params_per_phase = calculate_standard_params_per_phase(student_data_norm, dynamic_frame_mapping)

    # 3. Tính điểm cho từng nhịp
    final_scores = {}
    for phase in standard_frame_mapping.keys(): # Dùng key từ mapping gốc để đảm bảo đủ các nhịp
        if phase not in std_params_per_phase or phase not in student_params_per_phase:
            print(f"Cảnh báo: Bỏ qua nhịp '{phase}' do thiếu dữ liệu ở video chuẩn hoặc video của học sinh.")
            continue

        std_params = std_params_per_phase[phase]
        student_params = student_params_per_phase[phase]
        weights = phase_weights.get(phase, {})

        phase_score = calculate_phase_score(std_params, student_params, weights, difficulty_threshold)
        final_scores[phase] = phase_score

    print("Điểm số cuối cùng:", final_scores)
    return final_scores

def generate_frame_mapping_from_video(video_path, progress_callback=None):
    """
    Phân tích video để tự động tạo ra một frame_mapping gợi ý.
    Hàm này sẽ tìm các điểm chuyển động cực đại (đỉnh) và coi chúng là điểm kết thúc của mỗi nhịp.
    """
    if progress_callback:
        progress_callback(10, "Bắt đầu phân tích video mẫu...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(get_resource_path('models/yolov8s-pose.pt')) # Sử dụng cùng model với file này
    
    all_keypoints = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        if results[0].keypoints and results[0].keypoints.data.shape[0] > 0:
            # Lấy keypoints của người đầu tiên được phát hiện
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            all_keypoints.append(keypoints)
        else:
            all_keypoints.append(None)
        
        frame_idx += 1
        if progress_callback:
            progress = int((frame_idx / total_frames) * 70) if total_frames > 0 else 0
            progress_callback(10 + progress, f"Trích xuất keypoints... ({frame_idx}/{total_frames})")

    cap.release()

    if not all_keypoints:
        return {}

    if progress_callback:
        progress_callback(85, "Phân tích chuyển động...")

    # Tính toán sự thay đổi vị trí của một keypoint trung tâm (ví dụ: mũi)
    # Keypoint 0 là mũi (nose)
    nose_positions = [kp[0] if kp is not None and kp.shape[0] > 0 else None for kp in all_keypoints]
    
    velocities = []
    for i in range(1, len(nose_positions)):
        if nose_positions[i] is not None and nose_positions[i-1] is not None:
            dist = np.linalg.norm(nose_positions[i] - nose_positions[i-1])
            velocities.append(dist)
        else:
            velocities.append(0)

    peaks, properties = find_peaks(velocities, height=np.mean(velocities) * 0.8, distance=15)

    # --- Cải tiến: Giới hạn số nhịp là 8 ---
    # Nếu có nhiều hơn 7 đỉnh (tạo ra > 8 nhịp), chọn 7 đỉnh nổi bật nhất.
    if len(peaks) > 7:
        peak_heights = properties['peak_heights']
        # Lấy chỉ số của 7 đỉnh cao nhất
        top_peak_indices = np.argsort(peak_heights)[-7:]
        # Lấy các đỉnh tương ứng và sắp xếp lại theo thứ tự frame
        peaks = np.sort(peaks[top_peak_indices])

    if progress_callback:
        progress_callback(95, "Tạo phân chia nhịp...")

    frame_mapping = {}
    start_frame = 0
    for i, peak_frame in enumerate(peaks): # Vòng lặp này giờ sẽ chạy tối đa 7 lần
        phase_name = f"Nhịp {i + 1}"
        frame_mapping[phase_name] = [start_frame, int(peak_frame)]
        start_frame = int(peak_frame)

    # Đảm bảo nhịp cuối cùng được thêm vào và không vượt quá 8 nhịp
    if start_frame < total_frames - 1 and len(frame_mapping) < 8:
         phase_name = f"Nhịp {len(frame_mapping) + 1}"
         frame_mapping[phase_name] = [start_frame, total_frames - 1]

    return frame_mapping

def suggest_phase_weights(standard_data, frame_mapping):
    """
    Phân tích dữ liệu video mẫu để gợi ý trọng số cho từng nhịp.
    Trọng số được gợi ý dựa trên phương sai (variance) của các góc trong mỗi nhịp.
    Góc nào có phương sai lớn hơn (thay đổi nhiều hơn) sẽ có trọng số cao hơn.
    """
    print("-> Bắt đầu gợi ý trọng số cho các nhịp...")
    suggested_weights = {}
    
    # Các thông số quan tâm (giống trong UI)
    TARGET_METRICS = [
        "L_SHOULDER_ANGLE", "R_SHOULDER_ANGLE", "L_ELBOW_ANGLE", "R_ELBOW_ANGLE",
        "L_HIP_ANGLE", "R_HIP_ANGLE", "L_KNEE_ANGLE", "R_KNEE_ANGLE", "WRIST_DISTANCE", "ANKLE_DISTANCE"
    ]

    # 1. Tính toán tất cả các tham số cho mỗi frame
    # Preprocess first to handle missing frames
    processor = ReferencePoseProcessor()
    processed_data = processor.preprocess_keypoints(standard_data)
    
    all_frame_params = [calculate_params_for_frame(kps) for kps in processed_data]

    for phase_name, (start_frame, end_frame) in frame_mapping.items():
        end_frame = min(end_frame, len(all_frame_params))
        start_frame = min(start_frame, end_frame)
        
        phase_params_frames = all_frame_params[start_frame:end_frame]
        
        if not phase_params_frames:
            suggested_weights[phase_name] = {key: 0.0 for key in TARGET_METRICS}
            continue

        # 2. Thu thập giá trị của từng thông số trong nhịp
        param_values = {key: [] for key in TARGET_METRICS}
        for frame_p in phase_params_frames:
            if frame_p:
                for key in TARGET_METRICS:
                    if key in frame_p:
                        param_values[key].append(frame_p[key])

        # 3. Tính phương sai cho từng thông số
        param_variances = {key: float(np.var(values)) if len(values) > 1 else 0.0 for key, values in param_values.items()}
        
        # 4. Chuẩn hóa phương sai thành trọng số (tổng bằng 1)
        total_variance = sum(param_variances.values())
        phase_w = {key: 0.0 for key in TARGET_METRICS}
        if total_variance > 0:
            for key, variance in param_variances.items():
                phase_w[key] = round(float(variance / total_variance), 2)
        
        # 5. Điều chỉnh để đảm bảo tổng là 1 sau khi làm tròn
        current_sum = sum(phase_w.values())
        if not np.isclose(current_sum, 1.0) and current_sum > 0:
            diff = 1.0 - current_sum
            max_key = max((k for k, v in phase_w.items() if v > 0), key=phase_w.get, default=None)
            if max_key:
                phase_w[max_key] = round(phase_w[max_key] + diff, 2)

        suggested_weights[phase_name] = phase_w

    print("-> Gợi ý trọng số hoàn tất.")
    return suggested_weights
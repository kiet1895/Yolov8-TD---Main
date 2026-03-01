import cv2
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ultralytics import YOLO
import os
from .utils import get_resource_path

class ReferencePoseProcessor:
    def __init__(self, model_path='models/yolov8s-pose.pt'):
        # Use get_resource_path to work in both dev and bundled environment
        full_model_path = get_resource_path(model_path)
        self.model = YOLO(full_model_path)
        self.smoothing_window = 15
        self.polyorder = 2

    def process_video(self, video_path, output_path=None, use_ideal_rep=False):
        """
        Main pipeline:
        1. Extract Raw Keypoints
        2. Preprocess (Interpolate Missing -> Smooth)
        3. Normalize (Centering + Scaling) -> Crucial for alignment
        4. (Optional) Segment & Generate Ideal Rep
        """
        print(f"--> [Processor] Starting advanced processing for: {video_path}")
        
        # 1. Extract
        raw_keypoints, meta = self._extract_keypoints(video_path)
        if not raw_keypoints or len(raw_keypoints) < 30:
            print("--> [Processor] Video too short or no keypoints found.")
            return None

        # 2. Preprocess (Clean & Smooth)
        curated_kps = self.preprocess_keypoints(raw_keypoints)

        # 3. Normalize (Center to hip, Scale to torso length)
        normalized_kps = self.normalize_sequence(curated_kps)

        if use_ideal_rep:
            # 4. Detect Repetitions
            reps_indices = self._segment_repetitions(normalized_kps, meta['fps'])
            
            # 5. Generate Ideal Rep
            ideal_rep_kps = self._generate_ideal_rep(normalized_kps, reps_indices)
            print(f"--> [Processor] Ideal rep generated. Length: {len(ideal_rep_kps)} frames.")
            return ideal_rep_kps
        else:
             print(f"--> [Processor] returning full normalized sequence. Length: {len(normalized_kps)} frames.")
             return normalized_kps

    def normalize_sequence(self, keypoints):
        """
        Normalize a sequence of keypoints:
        - Center: Mid-hip to (0,0)
        - Scale: Torso length (mid-shoulder to mid-hip) = 1.0
        """
        norm_sequence = np.zeros_like(keypoints)
        # COCO Keypoints: 
        # 5,6 Shoulders; 11,12 Hips.
        
        for t in range(len(keypoints)):
            frame_kps = keypoints[t] # 17 x 3
            
            # --- Centering ---
            # Mid hip = (left_hip + right_hip) / 2
            l_hip = frame_kps[11, :2]
            r_hip = frame_kps[12, :2]
            mid_hip = (l_hip + r_hip) / 2.0
            
            # Shift all points
            frame_kps[:, :2] -= mid_hip
            
            # --- Scaling ---
            # Torso Size = Dist(Mid-Shoulder, Mid-Hip)
            l_sh = frame_kps[5, :2]
            r_sh = frame_kps[6, :2]
            mid_shoulder = (l_sh + r_sh) / 2.0
            
            # Since mid_hip is now at (0,0), mid_shoulder is just its coordinates
            torso_size = np.linalg.norm(mid_shoulder) 
            
            if torso_size > 0:
                scale_factor = 1.0 / torso_size
                frame_kps[:, :2] *= scale_factor
            
            norm_sequence[t] = frame_kps
            
        return norm_sequence

    def _extract_keypoints(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, verbose=False)
            if results[0].keypoints and results[0].keypoints.data.shape[0] > 0:
                # Take the first person found
                frames_list.append(results[0].keypoints.data[0].cpu().numpy())
            else:
                frames_list.append(None) # Mark missing frame
        
        cap.release()
        return frames_list, {'fps': fps}

    def preprocess_keypoints(self, raw_kps):
        """
        Fill missing frames using linear interpolation and apply Savitzky-Golay smoothing.
        """
        # Convert to numpy array (Time, Joints, 3) -> (T, 17, 3) (x, y, conf)
        # Handle None by temporary placeholder or masking
        valid_indices = [i for i, k in enumerate(raw_kps) if k is not None]
        if not valid_indices:
            return np.array([])

        # Create a contiguous array for interpolation
        T = len(raw_kps)
        num_joints = 17
        data_array = np.zeros((T, num_joints, 3))
        
        # Fill knowns
        for i in valid_indices:
            data_array[i] = raw_kps[i]

        # Interpolate missing frames
        for j in range(num_joints):
            for c in range(3): # x, y, conf
                y = data_array[valid_indices, j, c]
                x = valid_indices
                f = interp1d(x, y, kind='linear', fill_value="extrapolate")
                data_array[:, j, c] = f(np.arange(T))

        # Smooth (x, y) coordinates
        # Don't smooth confidence scores necessarily, but x,y are crucial
        smoothed_data = data_array.copy()
        for j in range(num_joints):
            for c in range(2): # Smooth only x and y
                smoothed_data[:, j, c] = scipy.signal.savgol_filter(
                    data_array[:, j, c], 
                    window_length=min(self.smoothing_window, T if T % 2 else T-1), 
                    polyorder=self.polyorder
                )
        
        return smoothed_data

    def _segment_repetitions(self, keypoints, fps):
        """
        Detect exercise reps based on movement velocity of key joints (Hands/Feet).
        Returns a list of (start_frame, end_frame) tuples.
        """
        # Calculate velocity of wrists and ankles combined
        # Keypoints: 9,10 (Wrists), 15,16 (Ankles)
        target_joints = [9, 10, 15, 16] 
        velocities = []
        
        valid_frames = keypoints[:, :, :2] # T x 17 x 2
        
        for t in range(1, len(valid_frames)):
            curr = valid_frames[t]
            prev = valid_frames[t-1]
            # Sum euclidean dist for target joints
            dist = 0
            for j in target_joints:
                dist += np.linalg.norm(curr[j] - prev[j])
            velocities.append(dist)
            
        velocities = np.array(velocities)
        
        # Smooth velocity to find reliable peaks (low pass filter)
        velocities_smooth = scipy.signal.savgol_filter(velocities, window_length=int(fps), polyorder=2)
        
        # Detect valleys (statistically low movement usually indicates start/end of a rep in hold poses)
        # OR detect periodic peaks. Ideally, we look for similar states.
        # Simplify: Use autocorr or find peaks in position if available.
        # Let's try finding peaks in velocity (max movement) and splitting by minima (stops).
        
        # This is heuristic. Ideally user provides one rep, but if we want "Ideal Rep", we assume repeated motion.
        # Finding minima in velocity (stops) is often a good delimiter.
        
        # Fallback: if not periodic enough, return whole video as 1 rep.
        
        prominence = np.mean(velocities_smooth) * 0.5
        peaks, _ = scipy.signal.find_peaks(-velocities_smooth, prominence=None) # Finding valleys by inverting
        
        # If we have distinct valleys, use them as split points
        if len(peaks) > 2:
            segments = []
            for i in range(len(peaks) - 1):
                segments.append((peaks[i], peaks[i+1]))
            return segments
        
        return [(0, len(keypoints))]

    def _generate_ideal_rep(self, keypoints, reps_indices):
        """
        Align all reps to the median length rep using DTW and average them.
        """
        if len(reps_indices) == 1:
            start, end = reps_indices[0]
            return keypoints[start:end]

        print(f"--> [Processor] Found {len(reps_indices)} repetitions. Averaging...")
        
        # 1. Extract individual segments
        reps_data = [keypoints[start:end] for (start, end) in reps_indices]
        
        # 2. Pick reference (median length)
        reps_data.sort(key=len)
        median_idx = len(reps_data) // 2
        ref_rep = reps_data[median_idx]
        
        # 3. Align others to reference
        aligned_reps = [ref_rep] # Include reference itself (weight 1)
        
        ref_flat = ref_rep[:, :, :2].reshape(len(ref_rep), -1) # Flatten for DTW
        
        for i, rep in enumerate(reps_data):
            if i == median_idx: continue
            
            rep_flat = rep[:, :, :2].reshape(len(rep), -1)
            dist, path = fastdtw(ref_flat, rep_flat, dist=euclidean)
            
            # Warp 'rep' to match 'ref_rep' timeline
            warped_rep = np.zeros_like(ref_rep)
            
            # path is list of (x, y) indices into ref and rep
            # Multiple ref indices might map to one rep index (compression) or vice versa (expansion)
            # We construct warped_rep[t] by averaging all rep frames that map to ref[t]
            
            path_dict = {} # ref_idx -> list of rep_indices
            for r_idx, q_idx in path:
                if r_idx not in path_dict: path_dict[r_idx] = []
                path_dict[r_idx].append(q_idx)
                
            for t in range(len(ref_rep)):
                matched_indices = path_dict.get(t, [])
                if matched_indices:
                    # Average the frames from the query rep
                    frames_to_avg = rep[matched_indices]
                    warped_rep[t] = np.mean(frames_to_avg, axis=0)
                else:
                    # interpolating or copying prev?
                    # DTW path usually covers all indices, but to be safe:
                    warped_rep[t] = warped_rep[t-1] if t > 0 else rep[0]
            
            aligned_reps.append(warped_rep)
            
        # 4. Average aligned reps
        ideal_rep = np.mean(np.array(aligned_reps), axis=0)
        
        return ideal_rep

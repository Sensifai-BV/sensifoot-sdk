import cv2
import mediapipe as mp
import numpy as np

class SensifootTracker:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """Initializes the MediaPipe Pose backend and EMA memory."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        
        # The 10 critical lower-limb joints
        self.TARGET_JOINTS = {
            23: "L_Hip",   24: "R_Hip",
            25: "L_Knee",  26: "R_Knee",
            27: "L_Ankle", 28: "R_Ankle",
            29: "L_Heel",  30: "R_Heel",
            31: "L_Toe",   32: "R_Toe"
        }
        
        # Memory for Exponential Moving Average (EMA) smoothing
        self.vis_history = {}

    def extract_features(self, frame):
        """
        Processes a single BGR frame. 
        Returns a list of 20 normalized (X, Y) features, or None if visibility is too low.
        """
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None, frame # Return the frame for drawing/UI purposes if needed

        landmarks = results.pose_landmarks.landmark
        
        # Calculate overall lower-limb visibility
        visibility_scores = [landmarks[idx].visibility for idx in self.TARGET_JOINTS.keys()]
        avg_visibility = sum(visibility_scores) / len(visibility_scores)
        
        # Gate: 0.2 to allow Side-View occlusion
        if avg_visibility < 0.2:
            return None, frame

        # Calculate Mid-Hip Anchor for normalization
        l_hip, r_hip = landmarks[23], landmarks[24]
        h_mid_x = (l_hip.x + r_hip.x) / 2.0
        h_mid_y = (l_hip.y + r_hip.y) / 2.0
        
        frame_features = []
        
        for idx in sorted(self.TARGET_JOINTS.keys()):
            lm = landmarks[idx]
            raw_vis = lm.visibility
            
            # EMA SMOOTHING: 70% Old Visibility + 30% New Visibility
            if idx not in self.vis_history:
                self.vis_history[idx] = raw_vis
            else:
                self.vis_history[idx] = (0.7 * self.vis_history[idx]) + (0.3 * raw_vis)
            
            smoothed_vis = self.vis_history[idx]
            
            # CONFIDENCE MASKING: Normalize to mid-hip and apply smoothed visibility
            norm_x = (lm.x - h_mid_x) * smoothed_vis
            norm_y = (lm.y - h_mid_y) * smoothed_vis
            
            frame_features.extend([norm_x, norm_y])
            
        return frame_features, frame

    def close(self):
        """Releases MediaPipe resources."""
        self.pose.close()

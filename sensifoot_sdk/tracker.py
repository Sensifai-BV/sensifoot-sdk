import cv2
import mediapipe as mp
import numpy as np

class SensifootTracker:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """Initializes the MediaPipe Pose backend and EMA memory."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
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

        # Skeleton connections between lower-limb joints only
        self.LOWER_LIMB_CONNECTIONS = [
            (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg: hip→knee→ankle→heel/toe
            (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg: hip→knee→ankle→heel/toe
            (23, 24),                                  # Hip bridge
        ]

        # Visual style constants
        self.JOINT_COLOR      = (0, 255, 180)    # Cyan-green for active joints
        self.BONE_COLOR       = (255, 200, 0)    # Amber for skeleton lines
        self.LABEL_COLOR      = (255, 255, 255)  # White for joint labels
        self.JOINT_RADIUS     = 8
        self.BONE_THICKNESS   = 2
        
        # Memory for Exponential Moving Average (EMA) smoothing
        self.vis_history = {}

    def _draw_skeleton_overlay(self, frame, landmarks):
        """
        Draws lower-limb skeleton bones and joint keypoints onto the frame.
        Only TARGET_JOINTS are drawn; upper body is intentionally excluded.
        """
        h, w = frame.shape[:2]

        # --- Draw bones (skeleton connections) ---
        for start_idx, end_idx in self.LOWER_LIMB_CONNECTIONS:
            lm_start = landmarks[start_idx]
            lm_end   = landmarks[end_idx]

            # Only draw if both endpoints have reasonable visibility
            if lm_start.visibility > 0.2 and lm_end.visibility > 0.2:
                pt1 = (int(lm_start.x * w), int(lm_start.y * h))
                pt2 = (int(lm_end.x   * w), int(lm_end.y   * h))
                cv2.line(frame, pt1, pt2, self.BONE_COLOR, self.BONE_THICKNESS, cv2.LINE_AA)

        # --- Draw joint keypoints and labels ---
        for idx, label in self.TARGET_JOINTS.items():
            lm = landmarks[idx]
            if lm.visibility > 0.2:
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Outer ring (dark border for contrast)
                cv2.circle(frame, (cx, cy), self.JOINT_RADIUS + 2, (0, 0, 0), -1, cv2.LINE_AA)
                # Inner filled dot
                cv2.circle(frame, (cx, cy), self.JOINT_RADIUS, self.JOINT_COLOR, -1, cv2.LINE_AA)

                # Joint label — offset slightly above the dot
                cv2.putText(
                    frame, label,
                    (cx + 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    self.LABEL_COLOR, 1, cv2.LINE_AA
                )

        return frame

    def draw_countdown(self, frame, seconds_remaining, gesture_name, status_text=""):
        """
        Draws a countdown timer and gesture prompt HUD on top of the frame.
        Call this from onboarding.py during the pre-capture wait period.

        Args:
            frame:             BGR frame to annotate (modified in-place).
            seconds_remaining: Integer seconds left (0 triggers GO! display).
            gesture_name:      Name of the gesture being captured.
            status_text:       Optional extra line (e.g. distance label).
        Returns:
            Annotated frame.
        """
        h, w = frame.shape[:2]

        # Semi-transparent dark banner across the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture_name}",
                    (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.72,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Optional status line (distance label etc.)
        if status_text:
            cv2.putText(frame, status_text,
                        (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (180, 180, 180), 1, cv2.LINE_AA)

        # Big countdown number or GO!
        if seconds_remaining > 0:
            count_str = str(seconds_remaining)
            color     = (0, 200, 255)   # Yellow-ish during countdown
        else:
            count_str = "GO!"
            color     = (0, 255, 120)   # Green on GO

        # Centre the countdown text
        font_scale, thickness = 3.2, 5
        (tw, th), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        cv2.putText(frame, count_str,
                    ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    color, thickness, cv2.LINE_AA)

        return frame

    def extract_features(self, frame):
        """
        Processes a single BGR frame. 
        Returns a list of 20 normalized (X, Y) features, or None if visibility is too low.
        The returned frame always has the lower-limb skeleton overlay drawn on it.
        """
        output_frame = frame.copy()
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None, output_frame  # Return the frame for drawing/UI purposes if needed

        landmarks = results.pose_landmarks.landmark

        # Draw skeleton overlay on every frame regardless of visibility gate
        output_frame = self._draw_skeleton_overlay(output_frame, landmarks)
        
        # Calculate overall lower-limb visibility
        visibility_scores = [landmarks[idx].visibility for idx in self.TARGET_JOINTS.keys()]
        avg_visibility = sum(visibility_scores) / len(visibility_scores)
        
        # Gate: 0.2 to allow Side-View occlusion
        if avg_visibility < 0.2:
            return None, output_frame

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
            
        return frame_features, output_frame

    def close(self):
        """Releases MediaPipe resources."""
        self.pose.close()

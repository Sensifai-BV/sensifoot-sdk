import cv2
import time
import numpy as np
from collections import deque
from sensifoot_sdk.tracker import SensifootTracker
from sensifoot_sdk.personalizer import SensifootPersonalizer

def capture_gesture_data(tracker, cap, gesture_name, reps=3, window_size=60):
    """Uses the tracker to record a specific number of frames for a gesture."""
    captured_windows = []
    
    print(f"\n🎬 Get ready to perform: {gesture_name}")
    time.sleep(2)
    
    for rep in range(reps):
        print(f"🟢 GO! Perform rep {rep + 1}...")
        buffer = deque(maxlen=window_size)
        
        # Simple capture loop (In a real app, you'd use your A-OD gates here)
        while len(buffer) < window_size:
            ret, frame = cap.read()
            if not ret: break
            
            features, output_frame = tracker.extract_features(frame)
            if features:
                buffer.append(features)
                
            cv2.imshow("SDK Calibration", output_frame)
            cv2.waitKey(1)
            
        # 1. Convert to numpy array
        xy_data = np.array(buffer, dtype=np.float32)
        
        # 2. Calculate the velocities to match your 40-dimensional TCN requirement
        vel_data = np.diff(xy_data, axis=0)
        vel_data = np.pad(vel_data, ((1, 0), (0, 0)), mode='symmetric')
        
        # 3. Stack to create the final 40-feature tensor
        full_window = np.hstack((xy_data, vel_data))
        captured_windows.append(full_window)
        
        print(f"✅ Rep {rep + 1} captured.")
        time.sleep(1)
        
    return captured_windows

def run_onboarding():
    # 1. Initialize the SDK modules
    tracker = SensifootTracker()
    personalizer = SensifootPersonalizer(base_model_path="best_model_TCN_PHASE1.pth")
    cap = cv2.VideoCapture(0)

    # 2. Capture data for the target gestures
    target_gestures = {1: "Toe Tap", 2: "Heel Raise"} # Add all 8 here
    
    for gesture_id, name in target_gestures.items():
        windows = capture_gesture_data(tracker, cap, name)
        personalizer.add_calibration_data(gesture_id, windows)

    # 3. Trigger the Hybrid Flash Training
    print("\n🚀 Starting edge personalization...")
    final_model_path = personalizer.flash_train_and_export()
    
    print(f"\n🎉 Success! Optimized ONNX engine saved to: {final_model_path}")
    
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_onboarding()
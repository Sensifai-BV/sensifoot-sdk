import cv2
import time
import numpy as np
from collections import deque
from sensifoot_sdk.tracker import SensifootTracker
from sensifoot_sdk.personalizer import SensifootPersonalizer

COUNTDOWN_SECONDS = 5  # How long to count down before each capture begins

def run_countdown(tracker, cap, gesture_name, status_text="", countdown=COUNTDOWN_SECONDS):
    """
    Displays a live countdown on the webcam feed using tracker.draw_countdown().
    The skeleton overlay is rendered on every frame during the wait period so the
    user can check their pose before the capture begins.

    Args:
        tracker:      SensifootTracker instance (provides draw_countdown + extract_features).
        cap:          OpenCV VideoCapture object.
        gesture_name: Name shown in the HUD banner.
        status_text:  Optional subtitle (e.g. distance label).
        countdown:    Number of seconds to count down (default: COUNTDOWN_SECONDS).
    """
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw skeleton even during countdown so the user can verify pose
        _, annotated = tracker.extract_features(frame)

        elapsed          = time.time() - start_time
        seconds_left     = max(0, countdown - int(elapsed))
        annotated        = tracker.draw_countdown(annotated, seconds_left, gesture_name, status_text)

        cv2.imshow("SDK Calibration", annotated)
        cv2.waitKey(1)

        if elapsed >= countdown:
            break


def capture_gesture_data(tracker, cap, gesture_name, reps=3, window_size=60):
    """
    Uses the tracker to record a specific number of gesture reps.
    Each rep is preceded by a 5-second live countdown so the user can
    prepare while still seeing their joint overlay on screen.
    """
    captured_windows = []
    
    print(f"\n🎬 Get ready to perform: {gesture_name}")

    for rep in range(reps):
        status = f"Rep {rep + 1} of {reps} — prepare!"
        print(f"\n⏳ {status} Countdown starting...")

        # ── 5-second live countdown with skeleton overlay ──
        run_countdown(tracker, cap, gesture_name, status_text=status)

        print(f"🟢 GO! Capturing rep {rep + 1}...")
        buffer = deque(maxlen=window_size)

        # ── Capture loop (A-OD gates can wrap this in production) ──
        while len(buffer) < window_size:
            ret, frame = cap.read()
            if not ret:
                break

            features, output_frame = tracker.extract_features(frame)

            # HUD: show "RECORDING" state during active capture
            cv2.putText(output_frame, "● RECORDING",
                        (16, output_frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 220), 2, cv2.LINE_AA)

            if features:
                buffer.append(features)

            cv2.imshow("SDK Calibration", output_frame)
            cv2.waitKey(1)

        # 1. Convert to numpy array
        xy_data = np.array(buffer, dtype=np.float32)

        # 2. Calculate velocities to match the 40-dimensional TCN requirement
        vel_data = np.diff(xy_data, axis=0)
        vel_data = np.pad(vel_data, ((1, 0), (0, 0)), mode='symmetric')

        # 3. Stack to create the final 40-feature tensor
        full_window = np.hstack((xy_data, vel_data))
        captured_windows.append(full_window)

        print(f"✅ Rep {rep + 1} captured.")
        time.sleep(0.5)     # Brief pause before next rep's countdown

    return captured_windows

def run_onboarding():
    # 1. Initialize the SDK modules
    tracker = SensifootTracker()
    personalizer = SensifootPersonalizer(base_model_path="./sensifoot_sdk/best_model_TCN_PHASE1.pth")
    cap = cv2.VideoCapture(0)

    # 2. Capture data for the target gestures
    target_gestures = {
        1: "Heel Tap", 2: "Forward Kick",
        3: "Foot Lift", 4: "Lateral Slide",
        5: "Forward Step", 6: "Cross Front",
        7: "Foot Hold", 8: "Flamingo Bend"
        }
    
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

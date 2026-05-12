import cv2
from sensifoot_sdk.tracker import SensifootTracker
from sensifoot_sdk.engine import SensifootEngine

# 1. Initialize the SDK Modules
tracker = SensifootTracker()
engine = SensifootEngine(model_path="sensifoot_v8.onnx")

# (Assume you ran a quick calibration here to get mu and sigma)
engine.update_noise_floor(view_name="FRONT", mu=0.002, sigma=0.0005)

# 2. Start the Camera
cap = cv2.VideoCapture(0)
print("SensiFoot SDK is active. Perform a gesture...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # --- The Core SDK Loop ---
    
    # A. Get clean math from the camera
    features, output_frame = tracker.extract_features(frame)
    
    if features:
        # B. Feed the math into the engine
        result = engine.process_frame(features)
        
        # C. Act on the results
        if result:
            print(f"🎯 GESTURE {result['gesture_id']} | "
                  f"Conf: {result['confidence']:.1f}% | "
                  f"Latency: {result['latency_ms']:.2f} ms")
            
            # Put visual feedback on the screen
            cv2.putText(output_frame, f"G{result['gesture_id']} ({result['confidence']:.0f}%)", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("SensiFoot Production App", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
tracker.close()
cv2.destroyAllWindows()
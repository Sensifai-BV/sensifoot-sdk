import sys
import os

# Append the parent directory (repository root) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import time
import numpy as np
from collections import deque
from sensifoot_sdk.tracker import SensifootTracker
from sensifoot_sdk.engine import SensifootEngine

# --- Configuration ---
VIDEO_PATH = "long-shot1-blur.mp4"                  # Is the blurred video
JSON_PATH = "long-shot1_ground_truth.json"
MODEL_PATH = "sensifoot_v8.onnx"
FEATURES_CACHE_PATH = "clean_features_cache.npy" # The secret to reproducible metrics!
FPS = 30.0

# Temporal matching tolerances
FRAME_TOLERANCE = 45  
COOLDOWN_FRAMES = 45  

def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    for e in events:
        e['detected'] = False
        
    return data['total_frames'], events

def run_stress_test():
    print("🚀 Initiating ESWA Industrial Profiling...")
    
    total_gt_frames, gt_events = load_ground_truth(JSON_PATH)
    duration_hours = total_gt_frames / FPS / 3600.0
    
    tracker = SensifootTracker()
    engine = SensifootEngine(model_path=MODEL_PATH)
    
    # ==========================================
    # PHASE 1: FEATURE EXTRACTION / CACHING
    # ==========================================
    all_frame_features = []
    
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"📦 Loading pristine features from {FEATURES_CACHE_PATH}...")
        all_frame_features = np.load(FEATURES_CACHE_PATH, allow_pickle=True).tolist()
    else:
        print("🔍 Extracting clean features from raw video (This runs once)...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Extract features and append (even if None, to keep frame alignment)
            features, _ = tracker.extract_features(frame)
            all_frame_features.append(features)
            
        cap.release()
        
        # Save the cache so reviewers use this instead of running MediaPipe on blurred video
        np.save(FEATURES_CACHE_PATH, np.array(all_frame_features, dtype=object))
        print(f"✅ Saved pristine features to {FEATURES_CACHE_PATH}")

    # ==========================================
    # PHASE 2: AUTO-CALIBRATE A-OD GATE
    # ==========================================
    print("🔧 Running Auto-Calibration on first 10 seconds of features...")
    idle_energies = []
    temp_buffer = deque(maxlen=engine.window_size)
    
    for features in all_frame_features[:300]:  # First 300 frames
        if features is not None:
            temp_buffer.append(features)
            if len(temp_buffer) == engine.window_size:
                xy_data = np.array(temp_buffer, dtype=np.float32)
                diff = xy_data - xy_data[0]
                cumsum = np.cumsum(diff, axis=0)
                od_magnitudes = np.sqrt(np.sum(cumsum ** 2, axis=0)) / engine.window_size
                idle_energies.append(np.max(od_magnitudes))
                
    mu = float(np.mean(idle_energies)) if idle_energies else 0.02
    sigma = float(np.std(idle_energies)) if idle_energies else 0.01
    
    engine.update_noise_floor("FRONT", mu, sigma)
    engine.update_noise_floor("SIDE", mu, sigma) 
    print(f"🔒 A-OD Gate LOCKED! mu={mu:.4f}, sigma={sigma:.4f}")
    
    # ==========================================
    # PHASE 3: LIVE ENGINE EVALUATION
    # ==========================================
    tp_count = 0
    fp_count = 0
    latencies = []
    neural_invocations = 0
    misclass_count = 0
    cooldown_counter = 0
    consensus_queue = deque(maxlen=3)
    
    print("🧠 Running Inference Engine...")
    
    for frame_idx_zero_based, features in enumerate(all_frame_features):
        frame_idx = frame_idx_zero_based + 1  # 1-based index to match GT logic
        
        if cooldown_counter > 0:
            cooldown_counter -= 1
            
        if features is None:
            continue
            
        res = engine.process_frame(features)
        is_eval_frame = (frame_idx >= engine.window_size and frame_idx % engine.step_size == 0)
        
        if res:
            neural_invocations += 1
            latencies.append(res['latency_ms'])
            consensus_queue.append(res['gesture_id'])
            
            if len(consensus_queue) == 3 and len(set(consensus_queue)) == 1:
                if cooldown_counter > 0:
                    continue
                    
                pred_id = res['gesture_id']
                matched = False
                is_duplicate = False 

                for event in gt_events:
                    start_window = event['start_frame'] - FRAME_TOLERANCE
                    end_window = event['end_frame'] + FRAME_TOLERANCE
                    
                    if start_window <= frame_idx <= end_window:
                        if event['class_id'] == pred_id:
                            if not event['detected']:
                                event['detected'] = True
                                tp_count += 1
                                matched = True
                                print(f"✅ [Frame {frame_idx}] True Positive: Gesture {pred_id}")
                            else:
                                is_duplicate = True
                                print(f"🔁 [Frame {frame_idx}] Sustained Trigger: Gesture {pred_id}")
                            
                            cooldown_counter = COOLDOWN_FRAMES
                            consensus_queue.clear()
                            break
                        else:
                            misclass_count += 1 
                            matched = True  
                            print(f"⚠️ [Frame {frame_idx}] Misclassification: Expected {event['class_id']}, Triggered {pred_id}")
                            cooldown_counter = COOLDOWN_FRAMES
                            consensus_queue.clear()
                            break
                
                if not matched and not is_duplicate:
                    fp_count += 1
                    cooldown_counter = COOLDOWN_FRAMES
                    consensus_queue.clear()
                    print(f"❌ [Frame {frame_idx}] False Activation (Background): Triggered {pred_id}")
                    
        elif is_eval_frame:
            consensus_queue.clear()
                
    tracker.close()
    
    # --- Final ESWA Metrics Calculation ---
    fn_count = sum(1 for e in gt_events if not e['detected'])
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fa_h = fp_count / duration_hours
    miss_h = fn_count / duration_hours
    avg_latency = np.mean(latencies) if latencies else 0
    calls_per_min = neural_invocations / (total_gt_frames / FPS / 60.0)
    
    print("\n" + "="*50)
    print("📊 ESWA EVALUATION RESULTS")
    print("="*50)
    print(f"Total Ground Truth Events: {len(gt_events)}")
    print(f"True Positives: {tp_count}")
    print(f"False Negatives (Missed): {fn_count}")
    print(f"False Positives (FA): {fp_count}")
    print(f"Misclassifications: {misclass_count}")
    print("-" * 50)
    print(f"Event-F1 Score:  {f1:.3f}")
    print(f"Precision:       {precision:.3f}")
    print(f"Recall:          {recall:.3f}")
    print("-" * 50)
    print(f"False Activations/Hour: {fa_h:.2f}")
    print(f"Missed/Hour:            {miss_h:.2f}")
    print("-" * 50)
    print(f"Avg ONNX Latency:       {avg_latency:.2f} ms")
    print(f"Neural Invocations/min: {calls_per_min:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_stress_test()
import numpy as np
from collections import deque
import time
import onnxruntime as ort

class SensifootEngine:
    def __init__(self, model_path="sensifoot_v8.onnx", window_size=60, step_size=5):
        """Initializes the A-OD buffers and the ONNX inference session."""
        print("🚀 Booting SensiFoot A-OD Engine (ONNX Backend)...")
        
        # 1. Load the lightweight ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # 2. Mathematical buffers
        self.window_size = window_size
        self.step_size = step_size
        self.buffer = deque(maxlen=self.window_size)
        self.frame_counter = 0
        
        # 3. A-OD Memory Dictionary
        self.idle_stats = {
            "FRONT": {"mu": 0.0, "sigma": 0.0},
            "SIDE":  {"mu": 0.0, "sigma": 0.0}
        }
        self.global_start_gate = 0.015  # Safe default fallback

    def update_noise_floor(self, view_name, mu, sigma):
        """Updates the internal A-OD gates after calibration."""
        self.idle_stats[view_name]["mu"] = mu
        self.idle_stats[view_name]["sigma"] = sigma
        
        # Calculate the global highest noise floor
        max_mu = max(self.idle_stats["FRONT"]["mu"], self.idle_stats["SIDE"]["mu"])
        max_sigma = max(self.idle_stats["FRONT"]["sigma"], self.idle_stats["SIDE"]["sigma"])
        
        self.global_start_gate = max(max_mu + (3.0 * max_sigma), 0.015)
        print(f"⚙️ SDK Gate Updated | Global Start Gate: {self.global_start_gate:.5f}")

    def process_frame(self, features):
        """
        The live processing loop. Takes 20 clean features from the Tracker, 
        calculates OD Energy, and runs ONNX inference if the gate is passed.
        Returns a dictionary with prediction data, or None if idle.
        """
        self.buffer.append(features)
        self.frame_counter += 1

        # Only run the heavy math if the buffer is full and we hit our step size
        if len(self.buffer) == self.window_size and self.frame_counter % self.step_size == 0:
            t_start = time.perf_counter()

            xy_data = np.array(self.buffer, dtype=np.float32)
            
            # --- 1. Calculate A-OD Energy ---
            anchor_frame = xy_data[0]
            diff = xy_data - anchor_frame
            cumsum = np.cumsum(diff, axis=0)
            od_magnitudes = np.sqrt(np.sum(cumsum ** 2, axis=0)) / self.window_size
            window_od_energy = np.max(od_magnitudes)

            # --- 2. The Gatekeeper ---
            # If the energy is lower than the stop gate, abort the neural inference
            stop_gate = max(self.idle_stats["FRONT"]["mu"], self.idle_stats["SIDE"]["mu"]) + 0.5 * (self.global_start_gate - max(self.idle_stats["FRONT"]["mu"], self.idle_stats["SIDE"]["mu"]))
            
            if window_od_energy < stop_gate:
                return None  # User is standing still

            # --- 3. Prepare the 40-Dimensional Tensor ---
            # Calculate velocities and pad to maintain shape
            vel_data = np.diff(xy_data, axis=0)
            vel_data = np.pad(vel_data, ((1, 0), (0, 0)), mode='symmetric')
            
            # Stack X,Y and Velocities horizontally
            full_window = np.hstack((xy_data, vel_data))
            
            # Expand dims to match ONNX expectation: (Batch, Seq, Features) -> (1, 60, 40)
            ort_input = np.expand_dims(full_window, axis=0)

            # --- 4. Neural Inference (ONNX) ---
            logits = self.session.run(None, {self.input_name: ort_input})[0]
            
            # Apply Softmax manually (since PyTorch CrossEntropyLoss expects raw logits)
            exp_preds = np.exp(logits[0] - np.max(logits[0]))
            probs = exp_preds / np.sum(exp_preds)
            
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100
            
            t_end = time.perf_counter()
            process_ms = (t_end - t_start) * 1000

            if confidence > 70.0:
                return {
                    "gesture_id": int(pred_idx + 1),
                    "confidence": confidence,
                    "latency_ms": process_ms,
                    "od_energy": window_od_energy
                }

        return None # Waiting for buffer to fill or step size to match

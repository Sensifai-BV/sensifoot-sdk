import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models_zoo import build_model
from export_sdk_model import export_v8_to_onnx  # The exporter script we made first

class SensifootPersonalizer:
    def __init__(self, base_model_path='./best_model_TCN_PHASE1.pth', device='cpu'):
        """Loads the base TCN model and prepares it for flash training."""
        self.device = torch.device(device)
        self.model = build_model(arch='TCN', input_dim=40, hidden_dim=32, num_classes=8).to(self.device)
        
        print(f"📦 Loading Base SensiFoot Model: {base_model_path}")
        self.model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        
        # Unfreeze network for maximum personalization
        for param in self.model.parameters():
            param.requires_grad = True

        self.captured_X = []
        self.captured_y = []

    def add_calibration_data(self, gesture_id, feature_windows):
        """
        Accepts recorded data windows for a specific gesture.
        Instead of the SDK handling the camera UI, the app developer passes the data here.
        """
        self.captured_X.extend(feature_windows)
        self.captured_y.extend([gesture_id - 1] * len(feature_windows))
        print(f"📥 Logged {len(feature_windows)} samples for Gesture {gesture_id}")

    def flash_train_and_export(self, batch_size=16, epochs=80, lr=1e-5):
        """
        Runs the State 2 training loop, saves the PyTorch weights, 
        and automatically compiles the final ONNX model for the Engine.
        """
        if len(self.captured_X) == 0:
            raise ValueError("No calibration data added! Cannot train.")

        print("\n🔥 Starting V8 Hybrid Flash Training...")
        X_tensor = torch.tensor(np.array(self.captured_X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(self.captured_y), dtype=torch.long)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch, lengths=None)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

        # Save the personalized PyTorch weights temporarily
        temp_pth_path = "temp_personalized.pth"
        torch.save(self.model.state_dict(), temp_pth_path)
        
        # ⚡ THE HYBRID BRIDGE ⚡
        # Instantly convert the newly trained model to ONNX for edge deployment
        final_onnx_path = "sensifoot_personalized_v8.onnx"
        export_v8_to_onnx(temp_pth_path, final_onnx_path)
        
        return final_onnx_path

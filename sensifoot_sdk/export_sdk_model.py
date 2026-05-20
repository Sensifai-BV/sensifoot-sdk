import torch
import onnx
import onnxruntime as ort
import numpy as np
from .models_zoo import build_model # Pulls your architecture

def export_v8_to_onnx(model_path, output_path, window_size=60, input_dim=40):
    print(f"🔄 Loading PyTorch model from {model_path}...")

    # 1. Initialize the model architecture (CPU is standard for export)
    device = torch.device('cpu')
    model = build_model(arch='TCN', input_dim=input_dim, hidden_dim=32, num_classes=8).to(device)
    
    # Load the personalized weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Create the dummy input tensor
    # Shape: (Batch Size, Sequence Length, Features) -> (1, 60, 40)
    # 40 features = 20 (X,Y) normalized EMA coordinates + 20 velocities
    dummy_input = torch.randn(1, window_size, input_dim, device=device)

    # 3. Trace and Export to ONNX
    print(f"📦 Compiling model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # Opset 14 is highly stable for mobile/CoreML translation
        do_constant_folding=True, # Optimizes static subgraphs
        input_names=['input_sequence'],
        output_names=['gesture_logits'],
        dynamic_axes={
            'input_sequence': {0: 'batch_size'}, # Preserves flexibility if you batch later
            'gesture_logits': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Successfully exported SensiFoot V8 to {output_path}")

    # 4. Verify graph integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("🔍 ONNX graph verified successfully.")
    
    # 5. Quick Inference Test
    session = ort.InferenceSession(output_path)
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    print(f"⚡ Test Output Shape: {ort_outs[0].shape} (Expected: 1, 8)")

if __name__ == "__main__":
    # Target the weights saved during your live pipeline flash-training
    INPUT_WEIGHTS = "parsa_live_personalized_tcn_v08.pth"
    OUTPUT_ONNX = "sensifoot_v8.onnx"
    
    export_v8_to_onnx(INPUT_WEIGHTS, OUTPUT_ONNX)

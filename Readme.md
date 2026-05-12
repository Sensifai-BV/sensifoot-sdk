# SensiFoot V8 Edge SDK

SensiFoot V8 is an industrial, edge-optimized software development kit (SDK) for real-time human lower-limb gesture recognition. 

Designed to process monocular RGB video feeds, this pipeline leverages MediaPipe Pose, Adaptive Ordinal Distance (A-OD) biomechanical gates, and a Temporal Convolutional Network (TCN). The SDK features a hybrid architecture: it performs on-device personalization using PyTorch (flash training in RAM), automatically compiles the personalized weights to ONNX format, and executes live inference via ONNX Runtime for near-zero latency on edge hardware.

## Core Innovations
* **Adaptive Ordinal Distance (A-OD) Gating:** A strictly mathematical $O(N)$ trigger system that calculates the environmental noise floor ($\mu + 3\sigma$) to autonomously separate user idle states from active gesture execution.
* **EMA-Smoothed Feature Extraction:** Extracts a highly optimized 40-dimensional feature set (20 spatial coordinates + 20 temporal velocities) using an Exponential Moving Average (EMA) visibility mask to prevent coordinate teleportation during occlusion.
* **Hybrid Edge Architecture:** Isolates heavy PyTorch gradient calculations to a one-time onboarding phase, allowing the live engine to run exclusively on lightweight ONNX runtimes.

## Repository Structure
.
├── main_temple.py                  # Production live-inference loop
├── onboarding.py                   # Full calibration and capture script
├── README.md                       
└── sensifoot_sdk/
    ├── __init__.py
    ├── best_model_TCN_PHASE1.pth   # Base PyTorch weights shipped with SDK
    ├── engine.py                   # ONNX inference module
    ├── export_sdk_model.py         # PyTorch-to-ONNX compiler
    ├── models_zoo.py               # TCN architecture definitions
    ├── personalizer.py             # Flash-training and data collection logic
    └── tracker.py                  # MediaPipe ingestion and EMA smoothing


## Installation

Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies:

git clone <YOUR_REPO_URL>
cd attempt014-sdk
pip install -r requirements.txt

*(Dependencies: `torch`, `onnx`, `onnxruntime`, `mediapipe`, `opencv-python`, `numpy`)*

---

## Usage Guide

The SensiFoot SDK supports two primary execution paths depending on whether you are calibrating a new user or deploying an already-calibrated model.

### Path A: Full User Calibration (New User)
Use this path if the engine needs to learn the specific biomechanical "Tolerance Cone" of a new user.

**1. Run the Onboarding Protocol:**

python onboarding.py

This script will:
* Activate the camera and calculate the A-OD noise floor.
* Prompt the user to perform target gestures to capture 60-frame feature windows.
* Flash-train the `best_model_TCN_PHASE1.pth` in RAM using the newly acquired data.
* Automatically compile and output a highly optimized `sensifoot_personalized_v8.onnx` file.

**2. Launch the Production Engine:**

python main_temple.py

This will boot the `SensifootEngine` using the newly compiled ONNX file, providing real-time gesture classification and latency metrics.

---

### Path B: Direct Deployment (Pre-Trained Model)
If you already possess a calibrated `.pth` model (e.g., `behzad_live_personalized_tcn_v08.pth`) and wish to bypass the onboarding phase, you can manually compile the SDK to use your existing weights.

**1. Compile the Existing Weights to ONNX:**
Open `sensifoot_sdk/export_sdk_model.py` and ensure the `INPUT_WEIGHTS` variable at the bottom points to your pre-trained `.pth` file. Then, run the exporter directly:

python sensifoot_sdk/export_sdk_model.py

This will instantly generate the `sensifoot_v8.onnx` file required by the edge engine.

**2. Launch the Production Engine:**
Ensure `main_temple.py` is pointing to your newly generated `.onnx` file, and start the live stream:

python main_temple.py


## Academic & Research Use
This architecture is currently under review for publication. If utilizing the A-OD gating mechanisms, the 40D feature extraction pipeline, or the SensiFoot TCN implementation, please refer to upcoming publications in *Engineering Application of Artificial Intelligence* / *Knowledge-Based Systems* for formal citation.
# SensiFoot V8 Edge SDK

SensiFoot V8 is an industrial, edge-optimized software development kit (SDK) for real-time human lower-limb gesture recognition. 

Designed to process monocular RGB video feeds, this pipeline leverages MediaPipe Pose, Adaptive Ordinal Distance (A-OD) biomechanical gates, and a highly optimized single LSTM network with a temporal attention layer. The SDK features a hybrid architecture: it performs on-device personalization using PyTorch (flash training in RAM), automatically compiles the personalized weights to ONNX format, and executes live inference via ONNX Runtime for near-zero latency on edge hardware.

## Core Innovations
* **Adaptive Ordinal Distance (A-OD) Gating:** A strictly mathematical $O(N)$ trigger system that calculates the environmental noise floor ($\mu + 3\sigma$) to autonomously separate user idle states from active gesture execution.
* **EMA-Smoothed Feature Extraction:** Extracts a highly optimized 36-dimensional feature set (Phase 5 configuration, including positional velocities and joint angles) using an Exponential Moving Average (EMA) visibility mask to improve resilience to monocular depth jitter and prevent coordinate teleportation during occlusion.
* **Hybrid Edge Architecture:** Isolates heavy PyTorch gradient calculations to a one-time onboarding phase, allowing the live engine to run exclusively on lightweight ONNX runtimes.

## Repository Structure
```text
.
├── experiments/                    # Academic benchmarking and stress tests
│   ├── run_eswa_stress_test.py     # ESWA evaluation script
│   ├── long-shot1-blur.mp4         # Anonymized visual reference video
│   ├── long-shot1_ground_truth.json# Event annotations
│   ├── clean_features_cache.npy    # Pre-extracted unblurred spatial features
│   └── benchmark.txt               # Evaluation logs
├── main_temple.py                  # Production live-inference loop
├── onboarding.py                   # Full calibration and capture script
├── README.md                       
└── sensifoot_sdk/
    ├── __init__.py
    ├── engine.py                   # ONNX inference module
    ├── export_sdk_model.py         # PyTorch-to-ONNX compiler
    ├── models_zoo.py               # LSTM architecture definitions
    ├── personalizer.py             # Flash-training and data collection logic
    └── tracker.py                  # MediaPipe ingestion and EMA smoothing
```

## Data and Code Availability (Model Weights)
To maintain a lightweight repository history, the heavy model weights are hosted on Hugging Face.

Download the weights here: Sensifai/Sensifoot-Edge-SDK

Before running the SDK, ensure you download the base .pth and .onnx files and place them in the root directory or configure your script paths accordingly.

## Installation
Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies:

```bash
git clone <YOUR_REPO_URL>
cd attempt014-sdk
pip install -r requirements.txt
```

(Dependencies: torch, onnx, onnxruntime, mediapipe, opencv-python, numpy)

## Academic Benchmarking & Reproducibility
To ensure strict reproducibility for academic review while maintaining ethical standards for human subjects, the experiments/ directory contains an offline stress-testing environment.

## Human Subject Privacy & Video Blurring
The visual reference video provided (experiments/long-shot1-blur.mp4) has been subjected to facial anonymization to comply with privacy standards. Because the MediaPipe Pose backbone relies on global body context to establish its bounding box, blurring facial pixels introduces a known, marginal spatial jitter that slightly degrades live tracking performance compared to the raw clinical baseline.

## Reproducing the Baseline Metrics
To allow reviewers to perfectly reproduce the exact baseline metrics reported in our upcoming manuscripts, we have decoupled the feature extraction from the inference engine. The repository includes clean_features_cache.npy, which contains the pristine, pre-extracted 36-dimensional feature set from the original, unblurred video.

To run the evaluation:

Navigate to the experiments folder:
```bash
cd experiments
```

Run the stress test:
```bash
python run_eswa_stress_test.py
```

The script will automatically detect the .npy cache, bypass the live camera tracker entirely, and feed the clean spatial coordinates directly into the SensiFoot Engine to compute the ESWA metrics (True Positives, False Activations/Hour, and Latency).

## Usage Guide
The SensiFoot SDK supports two primary execution paths depending on whether you are calibrating a new user or deploying an already-calibrated model.

### Path A: Full User Calibration (New User)
Use this path if the engine needs to learn the specific biomechanical "Tolerance Cone" of a new user.

#### 1. Run the Onboarding Protocol:

```bash
python onboarding.py
```

This script will:

- Activate the camera and calculate the A-OD noise floor.
- Prompt the user to perform target gestures to capture 60-frame feature windows.
- Flash-train the model in RAM using the newly acquired data.
- Automatically compile and output a highly optimized sensifoot_personalized_v8.onnx file.

#### 2. Launch the Production Engine:

```bash
python main_temple.py
```

This will boot the SensifootEngine using the newly compiled ONNX file, providing real-time gesture classification and latency metrics.

### Path B: Direct Deployment (Pre-Trained Model)

#### 1. Compile Existing Weights to ONNX:

Open `sensifoot_sdk/export_sdk_model.py` and set `INPUT_WEIGHTS` to your `.pth` file, then run:

```bash
python sensifoot_sdk/export_sdk_model.py
```

#### 2. Launch Production Engine:

Ensure `main_temple.py` points to the generated `.onnx` file and start the system.

## Academic & Research Use
This architecture is currently under review for publication. If utilizing the A-OD gating mechanisms, the Phase 5 feature extraction pipeline, or the SensiFoot LSTM implementation, please refer to upcoming publications in Engineering Application of Artificial Intelligence / Knowledge-Based Systems for formal citation.

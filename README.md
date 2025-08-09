# FER2013 Emotion Recognition (ResNet18/ResNet50)

A PyTorch project for facial expression recognition on FER2013. It provides custom ResNet18/ResNet50 backbones, configurable augmentation, warmup + LR schedulers, TensorBoard logging, checkpointing, and an optional webcam demo with saliency overlay.

## Features
- ResNet18/ResNet50 heads for 7 emotion classes
- AdaptiveAvgPool2d((1,1)) head (supports variable input sizes; match training size for best accuracy)
- Data augmentation: RandomResizedCrop, HorizontalFlip, Rotation
- SGD optimizer with momentum and weight decay (AdamW available)
- LR warmup + schedulers: CosineAnnealingLR or ReduceLROnPlateau
- TensorBoard logging, best-checkpoint saving
- Webcam demo with OpenCV DNN face detector, saliency visualization

## Model architectures
Number of trainable parameters in net50: 23516167
Number of trainable parameters in net18: 11173383
- ResNet18 head
  - Conv-BN-ReLU + MaxPool → 4 residual stages → AdaptiveAvgPool2d((1,1)) → Flatten → Linear(512→7)
- ResNet50 head
  - Conv-BN-ReLU + MaxPool → 4 bottleneck stages → AdaptiveAvgPool2d((1,1)) → Flatten → Dropout(0.5) → Linear(2048→7)

Note: While AdaptiveAvgPool2d allows variable inputs, inference should use the same input size used in training (e.g., 48×48 or 224×224) to avoid accuracy drop and excessive GPU memory usage.

## Data
- Dataset: FER2013 CSV (pixels are space-separated grayscale values)
- Default path: resnet/fer2013.csv
- Pass a custom path via CLI: --data-path path/to/fer2013.csv

## Key files
- train_cv.py
  - Model definitions: net18, net50
  - Training loop: train(...)
  - LR schedulers: warmup (LinearLR) + CosineAnnealingLR or ReduceLROnPlateau
  - Evaluation helper: eaccuracy_gpu(...)
  - Custom dataset class (fersets)
- test.py
  - Local CLI entry for training (argparse)
  - Loads FER2013 CSV, splits/builds DataLoaders, starts training
  - Writes TensorBoard logs, saves best checkpoint
- result.py
  - WebcamDemo: real-time inference with face detection
  - overlay_saliency_on_frame: saliency overlay on frames/ROIs
  - batch_attention_vis and evaluation utilities (confusion matrix, metrics)
- remote/fabfile.py
  - Fabric task run_train for remote training
- runs/ (generated)
  - TensorBoard logs per run
- weights/ (generated)
  - Best model weights per run

## Installation
- Python 3.9+ recommended
- Install dependencies:
  - PyTorch + torchvision (CUDA optional)
  - numpy, pandas, scikit-learn, tensorboard, opencv-python, pillow, tqdm
## Tipps
- For 4GB GPUs, start with --reshape 48 and --batch-size 8–32.
- Logs: runs/-net{NET}-lr{LR}-{TIMESTAMP}
- Weights: weights/-net{NET}-lr{LR}-{TIMESTAMP}/model_{NET}_lr{LR}.pth
- Best weights are saved when test accuracy improves.

Example (Windows, conda env):
````bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tensorboard opencv-python pillow tqdm

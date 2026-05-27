# VPIC — Venepuncture & IV Cannulation Object Detection

Part of the **VPIC AI-assisted AR/VR Training System** — a broader project building a real-time syringe and needle tracking pipeline for venepuncture and IV cannulation training on Meta Quest 3

This repository covers the **computer vision / object detection** component: dataset preparation, YOLOv8n model training, evaluation, and Unity integration for on-device AR inference.

---

## Project Overview

```
VPIC/
├── VPIC_DataPipeline.ipynb      # Data extraction, stats, visualisation, train/val/test split
├── VPIC_TrainEvalTest.ipynb     # Training, evaluation, test inference, ONNX export
├── requirements.txt             # Python dependencies
├── Unity/
│   ├── VPIC_YOLOv8Detector.cs  # Unity Sentis inference script (C#)
│   └── VPIC_Quest3_Setup.md    # Step-by-step Unity + Quest 3 setup guide
└── .gitignore
```

> `data/`, `vpic_dataset/`, `runs/`, and `venv/` are excluded from version control via `.gitignore` due to file size.

---

## System Architecture

```
Stereo Camera (Quest 3 Passthrough)
        │
        ▼
 Frame Preprocessing
  (640×640 resize, normalise)
        │
        ▼
  YOLOv8n ONNX Model
  (Unity Sentis, GPU inference)
        │
        ▼
 Detection Decoding + NMS
        │
        ▼
 AR Bounding Box Overlay
  (Screen Space Canvas, passthrough)
        │
        ▼
 Real-time Feedback to Trainee
```

---

## Dataset

- **Class:** `syringe` (single class, class ID 0)
- **Annotation format:** YOLO normalised `.txt` (cx cy w h)
- **Recording sessions:** 8 (`frames_1` through `frames_8`)
- **Total labelled pairs:** ~589 image + label pairs
- **Split:** 80% train / 10% val / 10% test (stratified by session)

| Split | Images |
|---|---|
| Train | 469 |
| Val | 53 |
| Test | 67 |

---

## Model Results

### Run 2 — GPU Training (Current)

Trained with **YOLOv8n** (nano) on 469 images for **44 epochs** (early stopping at patience 20) on **NVIDIA GeForce RTX 4050 Laptop GPU**. Total training time: ~4 minutes.

| Metric | Score |
|---|---|
| mAP@0.5 | **99.5%** |
| mAP@0.5:0.95 | **77.9%** |
| Precision | 98.8% |
| Recall | **100%** |
| Best epoch | 29 |
| Training time | ~4 min |
| Device | RTX 4050 Laptop GPU |

The model reached mAP@0.5 ≥ 0.99 by epoch 16 and maintained 100% recall throughout — no missed detections on the validation set. Early stopping triggered at epoch 44 (best at epoch 29, patience 20).

### Run Comparison

| Metric | Run 1 (CPU) | Run 2 (GPU) | Δ |
|---|---|---|---|
| mAP@0.5 | 99.4% | **99.5%** | +0.1% |
| mAP@0.5:0.95 | 81.9% | 77.9% | -4.0% |
| Precision | 97.3% | **98.8%** | +1.5% |
| Recall | 100% | 100% | — |
| Best epoch | 61 | **29** | -32 |
| Epochs trained | 81 | **44** | -37 |
| Training time | ~60 min | **~4 min** | 15× faster |

> mAP@0.5:0.95 is slightly lower in Run 2 due to fewer total epochs before early stopping — the model converged faster on GPU (best at epoch 29 vs 61) and stopped earlier. mAP@0.5 and recall are equivalent. This can be addressed by increasing `patience` or `epochs` in a future run.

---

## Training Findings

### Training Curves

![Training curves](docs/images/results.png)

All loss metrics (box, class, DFL) converge smoothly across train and val sets with no sign of overfitting. mAP@0.5 reaches ≥ 0.99 by epoch 16 and remains stable. Early stopping triggered at epoch 44.

### Precision-Recall Curve

![Precision-Recall curve](docs/images/BoxPR_curve.png)

Area under the PR curve = **0.995** — the model maintains near-perfect precision across the full recall range.

### F1 Curve

![F1 curve](docs/images/BoxF1_curve.png)

Peak F1 score at a confidence threshold of ~0.5, confirming the model is well-calibrated with a wide operating window.

### Confusion Matrix (Normalised)

![Confusion matrix](docs/images/confusion_matrix_normalized.png)

Near-perfect classification: 100% of syringe instances correctly detected, with background false positive rate effectively 0.

### Validation Predictions vs Ground Truth

| Ground Truth | Predictions |
|:---:|:---:|
| ![Ground truth labels](docs/images/val_batch0_labels.jpg) | ![Model predictions](docs/images/val_batch0_pred.jpg) |

Predictions are spatially tight and confidence scores are consistently high across varied syringe poses and backgrounds.

### Training Batch Sample

![Training batch](docs/images/train_batch0.jpg)

Sample of augmented training images with YOLO bounding box labels overlaid.

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/VPIC_ObjectDetection.git
cd VPIC_ObjectDetection

py -3.12 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# Install PyTorch with CUDA first (NVIDIA GPU required for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt
python -m ipykernel install --user --name=vpic --display-name "VPIC (venv)"
```

> **CPU fallback:** If you don't have an NVIDIA GPU, skip the torch CUDA install and run `pip install -r requirements.txt` directly. Training will use CPU (slower — ~60 min for 100 epochs on this dataset).

### 2. Add your data

Place your labelled frame sessions under:
```
data/
└── Syringe/
    ├── frames_1/
    │   ├── frames_1_frame_000001.png
    │   ├── frames_1_frame_000001.txt
    │   └── ...
    └── frames_2/ ... frames_8/
```

### 3. Run the data pipeline

Open `VPIC_DataPipeline.ipynb` in VS Code with the **VPIC (venv)** kernel and run all cells.
This generates `vpic_dataset/` with the YOLO-ready split and `dataset.yaml`.

### 4. Train and evaluate

Open `VPIC_TrainEvalTest.ipynb` and run all cells.
Training outputs land in `runs/vpic_syringe_v1/`. The final cell exports `best.onnx`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 training and ONNX export |
| `torch` / `torchvision` | PyTorch backend (CUDA 12.1 build recommended) |
| `opencv-python` | Image processing |
| `albumentations` | Training augmentation |
| `matplotlib` / `pandas` | Visualisation and metrics |
| `onnxruntime` | ONNX verification |
| Unity Sentis 2.x | On-device ML inference |
| Meta XR Core SDK | Quest 3 passthrough and XR |

See `requirements.txt` for pinned versions.

---

## Part of a Larger System

This repository is one component of the VPIC project:

| Component | Description |
|---|---|
| **Object Detection** *(this repo)* | YOLOv8n syringe tracking, Unity AR overlay |
| Stereo Camera Pipeline | Depth estimation, 3D needle position |
| React Dashboard | Real-time trainee feedback and session analytics |
| Haptic Feedback | Controller vibration on insertion events |

---

## Roadmap

- [x] GPU-accelerated training (RTX 4050, CUDA 12.1)
- [ ] Increase patience/epochs to recover mAP@0.5:0.95 (target >80%)
- [ ] Add `needle_tip` as a second detection class
- [ ] Stereo depth estimation for 3D syringe position
- [ ] WebSocket stream from Unity to React dashboard
- [ ] Retrain with `yolov8s` for higher mAP@0.5:0.95

---

## License

MIT

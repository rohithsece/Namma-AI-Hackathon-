# CROSSGUARD AI — Namma AI Hackathon

An edge-optimized machine learning pipeline for pedestrian crossing safety and vehicle monitoring, deploying **SSDLite320 MobileNetV3 Large** with **ExecuTorch FP32 (XNNPACK)** for low-latency embedded devices such as the Raspberry Pi 4.

[![Test Status](https://img.shields.io/badge/Unit%20Tests-6%2F6%20Passed-brightgreen)](#validation-and-benchmarks)
[![Parity](https://img.shields.io/badge/PyTorch%20%E2%86%92%20ExecuTorch-100%25%20Parity-blue)](#model-parity--export)
[![Latency](https://img.shields.io/badge/Inference%20Latency-8.82%20ms-orange)](#performance)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](crossguard_ml/LICENSE)

---

## 📄 Official Accuracy & Performance Report
Detailed evaluation benchmarks, floating-point parity analysis, and unit test verifications are compiled in the official PDF report:
* 👉 **[Download CROSSGUARD_AI_Test_Accuracy_Report.pdf](CROSSGUARD_AI_Test_Accuracy_Report.pdf)**

---

## 🎯 Key Features & Capabilities

- **Lightweight Pretrained Edge Detector:** SSDLite320 MobileNetV3 Large architecture with native COCO weights, pre-exported to ExecuTorch FP32 (`models/crossguard_ssdlite_fp32.pte`, ~13.85 MB).
- **6 Targeted Road Traffic Classes:** Monitors `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.
- **Per-Frame Instance Counting:** Computes per-class counts without requiring persistent cloud connections.
- **Safety Zone Heuristics:** Evaluates spatial co-occupancy between pedestrian crossing zones and vehicle approach zones to flag potential crossing hazards.
- **Offline & On-Device:** Fully self-contained offline execution; no cloud APIs or internet connectivity required during inference.

---

## 📊 Validation & Benchmark Summary

| Evaluation Domain | Metric | Result | Status |
|---|---|:---:|---|
| **Test Image 1 (`bus_india_cc0.jpg`)** | Bus Detection Confidence | **98.73%** | Detected, 0 False Positives (`8.30 ms`) |
| **Test Image 2 (`crosswalk_cc0.jpg`)** | Crossing Pedestrian #1 | **92.96%** | High-confidence spatial bounding box |
| | Crossing Pedestrian #2 | **80.03%** | Secondary pedestrian localized (`9.34 ms`) |
| **Pretrained Benchmark Baseline** | COCO Val2017 Box mAP (0.50:0.95) | **21.3%** | Official torchvision baseline (`AP50`: **36.7%**) |
| **Model Parity (PyTorch vs ExecuTorch)** | Functional Agreement | **100.0%** | Max box deviation $\le 0.000229\text{ px}$ |
| **Automated Test Suite** | Unit Tests Passed | **6 / 6 (100%)** | All NMS, scaling, and zone logic verified |
| **Inference Latency (Mean)** | Forward Execution Time | **8.82 ms** | **113.36 FPS** pure inference throughput |
| **End-to-End Pipeline Throughput** | Total Processing Speed | **27.54 FPS** | Includes image decode, NMS, annotation & write |

---

## 🚀 Quick Start Guide

### 1. Windows Setup & Run (PyTorch Preview)

Open PowerShell inside `crossguard_ml`:

```powershell
cd crossguard_ml

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 opencv-python-headless==5.0.0.93

# Run inference on bundled test images
python run.py --source samples --output outputs/my_run --backend torch --provenance public_sample
```

### 2. Raspberry Pi & Linux Setup (ExecuTorch Deployment)

On Raspberry Pi OS (64-bit) or Ubuntu Linux:

```bash
cd crossguard_ml
bash scripts/setup_linux.sh
source .venv/bin/activate

# Run inference on an organizer video clip
python run.py --source data/clip.mp4 --output outputs/pi_run --backend executorch --provenance organiser
```

---

## 📁 Repository Structure

```text
├── CROSSGUARD_AI_Test_Accuracy_Report.pdf   # Compiled accuracy & performance PDF
├── crossguard_ml/
│   ├── crossguard/                          # Core ML package
│   │   ├── model.py                         # SSDLite320 EdgeDetector graph wrapper
│   │   ├── pipeline.py                      # Preprocess, per-class NMS, zone analysis, annotations
│   │   └── dataset.py                       # COCO dataset loader & augmentation
│   ├── models/
│   │   ├── crossguard_ssdlite_fp32.pte      # Exported ExecuTorch FP32 XNNPACK model
│   │   └── ssdlite320_mobilenet_v3_large... # PyTorch checkpoint
│   ├── samples/                             # Bundled test imagery
│   │   ├── bus_india_cc0.jpg
│   │   └── crosswalk_cc0.jpg
│   ├── reports/                             # Measured latency, parity & sample outputs
│   ├── run.py                               # Main offline inference & counting runner
│   ├── evaluate.py                          # Labeled COCO mAP & MAE evaluation
│   ├── train.py                             # Optional fine-tuning script
│   ├── export_model.py                      # ExecuTorch lowering & export utility
│   ├── verify_model.py                      # Real-image parity verification script
│   └── tests/                               # Automated unit test suite
└── .gitignore
```

---

## 📜 License & Acknowledgments

- **Model Weights & Architecture:** SSDLite320 MobileNetV3 Large from torchvision (`BSD-3-Clause`).
- **Project Code:** Distributed under the terms of the [MIT License](crossguard_ml/LICENSE).

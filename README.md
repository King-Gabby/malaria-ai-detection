# 🔬 Malaria Parasite Detection System

> **AI-powered object detection of *Plasmodium vivax* parasites in blood smear microscopy images using YOLOv8.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-brightgreen.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Setup & Installation](#-setup--installation)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Interactive Demo](#-interactive-demo)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Project Overview

This system automates the detection and classification of malaria parasites in Giemsa-stained thin blood smear microscopy images. It identifies four life-cycle stages of *Plasmodium vivax* — **ring**, **trophozoite**, **schizont**, and **gametocyte** — alongside healthy red blood cells, providing:

- **Parasite localisation** with bounding boxes and confidence scores
- **Parasitemia estimation** (% infected cells) for severity assessment
- **Downloadable clinical reports** (PDF/CSV) for documentation
- **Interactive web demo** for real-time analysis

Built for the **University of Ibadan AI-in-Medicine Competition** with a focus on practical clinical applicability.

---

## 🏥 Problem Statement

Malaria kills over 600,000 people annually (WHO, 2023). Gold-standard diagnosis requires manual microscopy by trained technicians, a bottleneck in resource-limited settings where the disease burden is highest. Key challenges:

| Challenge | Impact |
|-----------|--------|
| Manual counting is slow (~20 min/slide) | Delayed treatment |
| Inter-observer variability | Inconsistent diagnoses |
| Shortage of trained microscopists | Limited access to diagnosis |
| Multiple parasite stages | Requires expert-level morphology knowledge |

**Our solution**: An AI system that detects and classifies parasites in seconds, providing consistent results to assist (no plans of replacing) human microscopists.

---

## 📊 Dataset

**[BBBC041 — Malaria Bounding Boxes](https://bbbc.broadinstitute.org/BBBC041)**
from the Broad Bioimage Benchmark Collection.

| Property | Value |
|----------|-------|
| Images | ~1,364 blood smear fields |
| Format | PNG, variable resolution |
| Annotations | JSON bounding boxes |
| Staining | Giemsa (thin smear) |
| Species | *Plasmodium vivax* |

### Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `ring` | Early trophozoite (ring form) — most common stage |
| 1 | `trophozoite` | Mature feeding stage |
| 2 | `schizont` | Replicative stage with merozoites |
| 3 | `gametocyte` | Sexual stage (transmissible to mosquitoes) |
| 4 | `red_blood_cell` | Healthy/uninfected RBC |



---

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────────┐
│  Raw Images │────▶│ Preprocessing│────▶│  YOLOv8    │────▶│  Streamlit   │
│  + JSON     │     │ (Stain Norm, │     │  Training   │     │  Demo App    │
│  Annotations│     │  CLAHE,      │     │  & Inference│     │  + Reports   │
└─────────────┘     │  Resize)     │     └────────────┘     └──────────────┘
                    └──────────────┘
```

**Model**: YOLOv8 (Ultralytics) with COCO pre-trained weights for transfer learning.

**Why YOLO?**
- Real-time inference speed (critical for demo day)
- Strong small-object detection (parasites are tiny relative to the field of view)
- One-stage detector → simpler deployment than Faster R-CNN pipelines
- Excellent community + documentation

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/King-Gabby/malaria-detection.git
cd malaria-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
python -m src.data.download_dataset --output_dir data/raw

# 5. Convert annotations to YOLO format
python -m src.data.convert_annotations \
    --annotations_json data/raw/malaria/training.json \
    --images_dir data/raw/malaria/images \
    --output_dir data/yolo_dataset \
    --val_split 0.15 \
    --test_split 0.15
```

> **Note:** Adjust the `--annotations_json` and `--images_dir` paths based on the
> actual extracted directory structure. Run `ls data/raw/` after download to verify.

---

## 🔄 Pipeline Walkthrough

### 1. Data Download & Conversion

```bash
# Download BBBC041 dataset
python -m src.data.download_dataset --output_dir data/raw

# Convert JSON annotations → YOLO .txt format with train/val/test split
python -m src.data.convert_annotations \
    --annotations_json data/raw/malaria/training.json \
    --images_dir data/raw/malaria/images \
    --output_dir data/yolo_dataset
```

### 2. (Optional) Preprocessing

```bash
# Apply stain normalisation and CLAHE contrast enhancement
python -m src.preprocessing.preprocess \
    --input_dir data/yolo_dataset/images/train \
    --output_dir data/processed/train \
    --normalize_stain \
    --apply_clahe \
    --target_size 640
```

> Preprocessing is optional because YOLOv8's built-in augmentations (mosaic, HSV jitter)
> already provide robustness. Use stain normalisation if your test images come from a
> different microscope/lab than the training data.

---

## 🏋️ Training

```bash
# Quick test (CPU, nano model, 10 epochs)
python -m src.training.train \
    --data configs/malaria.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --batch_size 8 \
    --device cpu

# Full training (GPU recommended)
python -m src.training.train \
    --data configs/malaria.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch_size 16 \
    --device 0 \
    --patience 15

# Resume interrupted training
python -m src.training.train \
    --data configs/malaria.yaml \
    --model runs/train/malaria_detection/weights/last.pt \
    --resume
```

**Model size guide:**

| Model | Params | Speed (CPU) | mAP (typical) | Use Case |
|-------|--------|-------------|----------------|----------|
| `nano` | 3.2M | ~100ms/img | Baseline | Quick iteration |
| `small` | 11.2M | ~200ms/img | Good | Balanced |
| `medium` | 25.9M | ~400ms/img | Better | Competition submission |
| `large` | 43.7M | ~700ms/img | Best | If you have GPU time |

> **CPU users:** Start with `nano` or `small`. Training will be slow (~hours) but functional.
> Use `--batch_size 4` if you run into memory issues.

---

## 📈 Evaluation

```bash
# Evaluate on test set
python -m src.evaluation.evaluate \
    --weights models/best.pt \
    --data configs/malaria.yaml \
    --output_dir results \
    --split test

# Evaluate on validation set
python -m src.evaluation.evaluate \
    --weights models/best.pt \
    --data configs/malaria.yaml \
    --split val
```

**Outputs** (saved to `results/`):
- `metrics.json` — mAP, precision, recall (overall + per-class)
- `confusion_matrix.png` — Detection confusion matrix heatmap
- `per_class_metrics.png` — Bar chart of AP50, precision, recall by class
- `summary_card.png` — Quick-reference metrics card

---

## 🖥 Interactive Demo

```bash
# Launch the Streamlit web app
streamlit run app/streamlit_app.py
```

**Features:**
- 📤 Upload any blood smear image
- 🎯 Real-time AI-powered detection with adjustable confidence threshold
- 📊 Parasitemia estimation with WHO severity classification
- 📄 Downloadable PDF report, CSV data, and annotated images
- 🎛 Toggle healthy RBC visibility for cleaner visualisation

---

## 📊 Results

> ⏳ **Placeholder** — > ⚠️ Baseline results below are from a 5-epoch test run on the single-class 
> (ring-only) label set, used to verify the training pipeline. Full 5-class, 
> 50-epoch results will be updated here once training completes.

| Metric | Value |
|--------|-------|
| mAP@0.5 |  98.9%  |
| mAP@0.5:0.95 | 81.0 |
| Precision (avg) |  96.1%  |
| Recall (avg) | 97.8 |

**Live Demo:** [malaria-ai-detection.streamlit.app](https://malaria-ai-detection.streamlit.app/)


### Per-Class Performance

| Class | AP@0.5 | Precision | Recall |
|-------|--------|-----------|--------|
| Ring | — | — | — |
| Trophozoite | — | — | — |
| Schizont | — | — | — |
| Gametocyte | — | — | — |
| Red Blood Cell | — | — | — |

---

## 📁 Project Structure

```
malaria-detection/
├── data/
│   ├── raw/                        # BBBC041 download (gitignored)
│   ├── processed/                  # Preprocessed images (gitignored)
│   └── yolo_dataset/               # YOLO-formatted dataset (gitignored)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── src/
│   ├── data/
│   │   ├── download_dataset.py     # Automated dataset download
│   │   └── convert_annotations.py  # BBBC041 JSON → YOLO .txt
│   ├── preprocessing/
│   │   └── preprocess.py           # Stain normalisation, CLAHE, resize
│   ├── training/
│   │   └── train.py                # YOLOv8 training script
│   ├── evaluation/
│   │   └── evaluate.py             # Metrics, confusion matrix, plots
│   └── inference/
│       └── predict.py              # Single/batch inference API
├── app/
│   └── streamlit_app.py            # Interactive web demo
├── models/                         # Saved weights (gitignored)
├── configs/
│   └── malaria.yaml                # YOLO dataset config
├── notebooks/                      # EDA, experimentation
├── tests/
│   ├── test_convert_annotations.py # Annotation conversion tests
│   └── test_preprocess.py          # Preprocessing pipeline tests
├── results/                        # Evaluation outputs
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
└── .gitignore
```

---

## 👥 Team

| Name | Role |
|------|------|
| Gabriel Akoleaje | Project Lead / Model Training / Data Pipeline / Preprocessing |
| Treasure Olajide | Streamlit UI / Demo |
| Sodiq Gbadegesin | Evaluation / Documentation |

**Competition:** NACOS UI × DATICAN 2026

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

### Dataset License
The BBBC041 dataset is provided by the Broad Institute under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Images from: Hung & Bhatt, *Determining Parasites in Giemsa-stained Thick Blood Smears* (BBBC).

---

## 🙏 Acknowledgements

- [Broad Bioimage Benchmark Collection (BBBC)](https://bbbc.broadinstitute.org/) for the BBBC041 dataset
- [Ultralytics](https://docs.ultralytics.com/) for the YOLOv8 framework
- World Health Organization — malaria diagnostic guidelines
- University AI-in-Medicine program faculty and mentors

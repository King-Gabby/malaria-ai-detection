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

This system automates the detection and classification of malaria parasites in 
Giemsa-stained thin blood smear microscopy images. It identifies four life-cycle 
stages of *Plasmodium vivax* — ring, trophozoite, schizont, and gametocyte — 
alongside healthy red blood cells, providing:

- **Parasite localisation** with bounding boxes and confidence scores
- **Parasitemia estimation** (% infected cells) for severity assessment, 
  following WHO treatment guideline thresholds
- **Uncertainty flagging** — detections in a low-confidence range are marked 
  for mandatory human review rather than auto-classified, respecting clinical 
  safety protocols
- **Batch processing mode** for analysing multiple slides at once, with a 
  downloadable summary table
- **Downloadable clinical reports** (PDF/CSV) for documentation
- **Interactive web demo** for real-time analysis

Built for the NACOS UI × DATICAN 2026 AI-in-Medicine Competition with a focus 
on practical clinical applicability and responsible AI deployment.
---

## 🏥 Problem Statement

Malaria kills over 600,000 people annually (WHO, 2023). Gold-standard diagnosis 
requires manual microscopy by trained technicians, a bottleneck in 
resource-limited settings where the disease burden is highest. Key challenges:

| Challenge | Impact |
|---|---|
| Manual counting is slow (~20 min/slide) | Delayed treatment |
| Inter-observer variability | Inconsistent diagnoses |
| Shortage of trained microscopists | Limited access to diagnosis |
| Multiple parasite stages | Requires expert-level morphology knowledge |

**Our solution:** An AI system that detects and classifies parasites in 
seconds, providing consistent, explainable results to *assist* — not replace — 
human microscopists. Where the model is uncertain, it says so explicitly rather 
than forcing a confident-looking but unreliable classification.

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

**Live version:** [malaria-ai-detection.streamlit.app](https://malaria-ai-detection.streamlit.app/)

**Features:**
- 📤 Upload a blood smear image, or try one of three bundled sample images
- 🎯 Real-time AI-powered detection with adjustable confidence and NMS thresholds
- ⚠️ Uncertainty flagging — low-confidence detections are highlighted and 
  marked for human verification, rather than silently auto-classified
- 📦 Batch processing mode — analyse multiple slides at once, with a 
  downloadable per-slide summary table
- 🔬 Detection close-up gallery — zoomed crops of each detected parasite 
  for visual verification
- 📊 Parasitemia estimation with WHO-guideline severity classification
- 🩺 AI Screening Summary card with a dynamic clinical recommendation
- 📄 Downloadable PDF report, CSV data, and annotated images
- 🎛 Toggle healthy RBC visibility for cleaner visualisation
- ⏱️ Live inference timing displayed per analysis
---

---

---

## 📊 Results

> ⏳ **In Progress** — Numbers below are from an interrupted 5-class training 
> run, checkpoint at epoch 35 of a planned 50. Training is resuming and 
> final results will be updated here once complete. Earlier results from a 
> single-class (ring-only) baseline run, used only to verify the pipeline, 
> have been superseded by this 5-class checkpoint.

| Metric | Value (epoch 35/50, 5-class, in progress) |
|--------|---------------------------------------------|
| mAP@0.5 | 59.5% |
| mAP@0.5:0.95 | 50.3% |
| Precision (avg) | 55.9% |
| Recall (avg) | 60.8% |

**Live Demo:** [malaria-ai-detection.streamlit.app](https://malaria-ai-detection.streamlit.app/)

### Per-Class Performance

> Per-class breakdown will be generated via `evaluate.py` once full training 
> completes. The current checkpoint shows strong performance on `ring` 
> specifically, with the remaining four classes (especially `schizont` and 
> `gametocyte`, the rarest classes in the dataset) expected to improve 
> substantially with the remaining training epochs.

| Class | AP@0.5 | Precision | Recall |
|-------|--------|-----------|--------|
| Ring | — | — | — |
| Trophozoite | — | — | — |
| Schizont | — | — | — |
| Gametocyte | — | — | — |
| Red Blood Cell | — | — | — |


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
malaria-ai-detection/
├── data/
│   └── raw/malaria/                # BBBC041 dataset, downloaded via kagglehub (gitignored)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/  # Normalised YOLO-format labels
├── src/
│   ├── data/
│   │   ├── download_dataset.py     # Dataset download utilities
│   │   └── convert_annotations.py  # BBBC041 JSON → YOLO .txt (alternate source path)
│   ├── preprocessing/
│   │   └── preprocess.py           # Stain normalisation, CLAHE, resize
│   ├── training/
│   │   └── train.py                # YOLOv8 training script, supports --resume
│   ├── evaluation/
│   │   └── evaluate.py             # Metrics, confusion matrix, plots
│   └── inference/
│       └── predict.py              # Single/batch inference API, uncertainty tiers
├── app/
│   ├── streamlit_app.py            # Interactive web demo
│   └── samples/                    # Bundled sample images for the demo
├── models/
│   └── best.pt                     # Trained weights (committed for deployment)
├── configs/
│   └── malaria.yaml                # YOLO dataset config
├── normalize_labels.py             # One-off script: converts pixel-coordinate 
│                                    # annotations to normalised YOLO format
├── notebooks/                      # EDA, experimentation
├── tests/
│   ├── test_convert_annotations.py
│   └── test_preprocess.py
├── requirements.txt                # Pinned dependencies
├── README.md
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
The BBBC041 dataset is provided by the Broad Institute under 
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). 
Images from: Hung & Bhatt, *Determining Parasites in Giemsa-stained Thick 
Blood Smears* (BBBC). Note: this dataset's non-commercial license applies to 
the underlying data and trained model weights; this repository's MIT license 
covers the original source code only.
---

## 🙏 Acknowledgements

- [Broad Bioimage Benchmark Collection (BBBC)](https://bbbc.broadinstitute.org/) for the BBBC041 dataset
- [Ultralytics](https://docs.ultralytics.com/) for the YOLOv8 framework
- World Health Organization: malaria diagnostic guidelines
- Mentors: Professor Onifade. Professor Akinola. All other Computer & Medical Lecturers and staffs of University of Ibadan, Oyo State.
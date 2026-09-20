# 🔬 RaphaID AI

### Offline Multi-Disease Diagnostic Tool for African Healthcare

RaphaID AI is an offline, CPU-optimized clinical decision-support platform designed for lab technicians, scientists, and doctors in Nigerian and African hospitals with inadequate specialist personnel. It combines three integrated modules—**Detection** (blood pathology), **Radiology** (imaging), and **Chatbot** (RAG-powered medical assistant)—into a single lightweight application that runs entirely on 8GB RAM laptops without internet connectivity.

---

<!-- Badges -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CPU Optimized](https://img.shields.io/badge/CPU-Optimized-success)](https://github.com/King-Gabby/malaria-ai-detection)
[![Offline First](https://img.shields.io/badge/Offline-First-blueviolet)](https://github.com/King-Gabby/malaria-ai-detection)

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Architecture](#-architecture)
- [Three Modules](#-three-modules)
- [Technology Stack](#-technology-stack)
- [Setup & Installation](#-setup--installation)
- [Model Optimization](#-model-optimization)
- [Session Management](#-session-management)
- [Repository Structure](#-repository-structure)
- [Roadmap](#-roadmap)
- [Team](#-team)
- [License](#-license)

---

## ✨ Highlights

* **Three Integrated Modules**: Detection (4 diseases), Radiology (3 modalities), Chatbot (RAG + LangGraph)
* **Fully Offline**: No cloud dependency, all models run locally on CPU
* **8GB RAM Optimized**: Quantized INT8/INT4 models, lazy loading, ONNX Runtime
* **Clinical Teal/Emerald UI**: Professional medical-grade dark theme
* **Session-Based**: User info, history, analytics stored locally (SQLite/JSON)
* **Human-in-the-Loop**: Clinician verification for uncertain detections
* **PDF/CSV Reports**: Clinical-grade reports with WHO severity classification

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAPHAID AI                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  DETECTION   │  │  RADIOLOGY   │  │  CHATBOT     │       │
│  │  (Blood)     │  │  (Imaging)   │  │  (RAG/LG)    │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                   │
│              ┌─────────────────────────┐                      │
│              │   LANGGRAPH ORCHESTRATOR │                     │
│              │  (Routes queries, calls  │                     │
│              │   models, manages RAG)   │                     │
│              └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
User Input → Module Selection → Sub-Module → Model Inference → 
Results + Confidence → Human Verification (if needed) → 
Session Storage → Analytics Update → Report Generation
```

---

## 🎯 Three Modules

### 1. Detection (Blood Pathology)
| Disease | Model | Classes | Dataset |
|---------|-------|---------|---------|
| **Malaria** | YOLOv8n | Ring, Trophozoite, Schizont, Gametocyte, RBC | BBBC041 |
| **Sickle Cell Disease** | YOLOv8n | Normal RBC, Sickle Cell, Target Cell, Spherocyte | Custom |
| **ALL (Acute Lymphoblastic Leukemia)** | YOLOv8n | Lymphoblast, Normal Lymphocyte, Smudge Cell | C-NMC / ASH |
| **Iron Deficiency** | YOLOv8n | Microcyte, Hypochromic, Normal RBC, Pencil Cell | Custom |

**Workflow**: Upload smear → Quality check → YOLOv8 detection → Uncertainty flagging → Clinician verify → Parasitemia/Count → WHO severity → PDF/CSV report

### 2. Radiology (Medical Imaging)
| Modality | Model | Classes | Dataset |
|----------|-------|---------|---------|
| **MRI** | YOLOv8n / UNet | Tumor, Edema, Normal Tissue | BraTS / fastMRI |
| **CT Scan** | YOLOv8n | Nodule, Consolidation, Effusion, Normal | LIDC / COVID-CT |
| **X-ray** | YOLOv8n | Pneumonia, TB, Cardiomegaly, Normal | CheXpert / NIH |

**Workflow**: Upload DICOM/PNG → Quality check → Detection/Segmentation → Radiologist verify → Measurements → Structured report

### 3. Chatbot (Medical Assistant)
* **RAG Pipeline**: Local vector store (ChromaDB/FAISS) with medical guidelines (WHO, NCDC, clinical protocols)
* **LangGraph Orchestrator**: Routes queries → Retrieves context → Generates grounded responses
* **Capabilities**: Differential diagnosis aid, treatment guidelines, drug interactions, lab interpretation
* **Guardrails**: Citation required, uncertainty flags, escalation triggers

---

## 💻 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Orchestrator** | LangGraph | Multi-agent coordination, RAG routing |
| **UI** | Streamlit + Custom CSS/HTML | Clinical dark theme, responsive layout |
| **Detection Models** | YOLOv8n (quantized) | Real-time object detection |
| **Radiology Models** | YOLOv8n / UNet (quantized) | Detection + Segmentation |
| **Embeddings** | sentence-transformers (quantized) | RAG vector search |
| **Vector Store** | ChromaDB / FAISS | Local document retrieval |
| **CV** | OpenCV | Image processing, quality checks |
| **Reports** | FPDF2 | Clinical PDF generation |
| **Analytics** | Plotly | Interactive charts, gauges |
| **Session** | SQLite / JSON | Local persistence |
| **Language** | Python 3.10+ | Core application logic |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- Git
- No GPU required (CPU-only deployment)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/King-Gabby/malaria-ai-detection.git
cd malaria-ai-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights (separate per disease/modality)
# Models are lazy-loaded on first use
# Place .pt/.onnx files in models/ directory

# 5. Run the application
streamlit run app/streamlit_app.py
```

### Model Files Structure
```
models/
├── detection/
│   ├── malaria_yolov8n.pt
│   ├── sickle_cell_yolov8n.pt
│   ├── all_yolov8n.pt
│   └── iron_deficiency_yolov8n.pt
├── radiology/
│   ├── mri_yolov8n.pt
│   ├── ct_yolov8n.pt
│   └── xray_yolov8n.pt
└── chatbot/
    ├── embeddings_model.onnx
    └── llm_model.gguf (optional, for local LLM)
```

---

## 🔬 Model Optimization Strategy

All models are optimized for **8GB RAM CPU-only deployment**:

| Technique | Implementation |
|-----------|----------------|
| **Quantization** | INT8/INT4 via ONNX Runtime / GGUF |
| **Model Pruning** | Structured pruning (30-50% params removed) |
| **Knowledge Distillation** | Teacher (YOLOv8m) → Student (YOLOv8n) |
| **ONNX Runtime** | CPU-optimized inference sessions |
| **Lazy Loading** | Models loaded only when module selected |
| **Separate Files** | One model file per disease/modality |
| **Memory Mapping** | mmap for large model weights |

**Target Metrics:**
- Model load time: < 3 seconds
- Inference time: < 500ms/image (CPU)
- Peak RAM usage: < 6GB (with all modules available)
- Model size: < 50MB each (quantized)

---

## 🗄 Session Management

* **In-Memory**: Streamlit `session_state` for active session
* **Persistent**: SQLite database (`~/.raphaid/sessions.db`) for history
* **Local Analytics**: JSON cache (`~/.raphaid/analytics.json`) for metrics
* **No Cloud**: Zero external dependencies, fully air-gapped capable

**Stored Data:**
- User preferences (theme, thresholds, defaults)
- Session history (patient IDs, results, timestamps)
- Verification decisions (accept/reject per detection)
- Aggregate metrics (detections per disease, accuracy trends)

---

## 📁 Repository Structure

```
raphaid-ai/
├── app/
│   ├── streamlit_app.py          # Main entry point
│   ├── components/
│   │   ├── __init__.py
│   │   ├── ui.py                 # Shared UI components
│   │   ├── theme.py              # Clinical Teal/Emerald theme
│   │   ├── navigation.py         # 3-module dropdown navigation
│   │   └── footer.py             # Session/Metrics/Analytics panels
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── malaria.py
│   │   │   ├── sickle_cell.py
│   │   │   ├── all_leukemia.py
│   │   │   └── iron_deficiency.py
│   │   ├── radiology/
│   │   │   ├── __init__.py
│   │   │   ├── mri.py
│   │   │   ├── ct_scan.py
│   │   │   └── xray.py
│   │   └── chatbot/
│   │       ├── __init__.py
│   │       ├── rag_pipeline.py
│   │       ├── langgraph_orchestrator.py
│   │       └── ui.py
│   └── samples/
│       └── .gitkeep
├── models/                       # Quantized model weights (gitignored)
├── src/
│   ├── inference/
│   │   ├── predict.py            # Base prediction logic
│   │   └── model_loader.py       # Lazy loading, quantization
│   ├── rag/
│   │   ├── vector_store.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   ├── orchestrator/
│   │   └── langgraph_orchestrator.py
│   └── utils/
│       ├── quality_check.py
│       ├── report_generator.py
│       └── session_manager.py
├── data/
│   ├── medical_knowledge/        # WHO guidelines, protocols
│   └── vector_index/             # ChromaDB/FAISS index
├── configs/
│   ├── detection.yaml
│   ├── radiology.yaml
│   └── chatbot.yaml
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

## 🚀 Roadmap

### Phase 1: Core UI & Navigation ✓
- [x] Clinical Teal/Emerald theme
- [x] 3-module navigation with dropdowns
- [x] Splash screen with RaphaID branding
- [x] Session info / Metrics / Analytics footer panels

### Phase 2: Detection Module (In Progress)
- [ ] Malaria detection pipeline (YOLOv8n + BBBC041)
- [ ] Sickle Cell detection pipeline
- [ ] ALL detection pipeline
- [ ] Iron Deficiency detection pipeline
- [ ] Quality checks + Human verification + Reports

### Phase 3: Radiology Module
- [ ] MRI tumor detection/segmentation
- [ ] CT scan nodule detection
- [ ] X-ray pathology detection
- [ ] DICOM support + Measurements

### Phase 4: Chatbot Module
- [ ] RAG pipeline with medical knowledge base
- [ ] LangGraph orchestrator skeleton
- [ ] Local LLM integration (GGUF/llama.cpp)
- [ ] Guardrails & citation system

### Phase 5: Integration & Optimization
- [ ] End-to-end testing on 8GB RAM laptop
- [ ] Model quantization pipeline (INT8/INT4)
- [ ] Performance profiling & optimization
- [ ] Documentation & deployment guides

---

## 👥 Team

| Name | Role | Links |
|------|------|-------|
| Gabriel Akoleaje | Project Lead / Model Training / Data Pipeline / Inference | [GitHub](https://github.com/King-Gabby) • [LinkedIn](https://www.linkedin.com/in/gabriel-akoleaje/) |
| Treasure Olajide | Streamlit UI / Demo / Clinical Workflow / UX | [GitHub](https://github.com/Strikertee) • [LinkedIn](https://www.linkedin.com/in/treasure-olajide-9a4963419/) |
| Sodiq Gbadegesin | Evaluation / Documentation / Testing | — |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- World Health Organization — Malaria, Sickle Cell, Leukemia, Anemia guidelines
- Nigeria Centre for Disease Control (NCDC) — Clinical protocols
- Broad Bioimage Benchmark Collection (BBBC) — BBBC041 Malaria dataset
- Ultralytics — YOLOv8 framework
- LangChain / LangGraph — Orchestration framework
- sentence-transformers — Embedding models
- ChromaDB / FAISS — Vector search
- All open-source medical AI researchers and clinicians
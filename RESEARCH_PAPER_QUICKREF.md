# RaphaID AI — Research Paper Quick Reference

## One-Paragraph Summary
**RaphaID AI** is an offline, CPU-optimized clinical decision-support platform integrating three diagnostic modules—**Detection** (4 blood diseases: Malaria, Sickle Cell, ALL, Iron Deficiency), **Radiology** (3 modalities: MRI Brain, CT Chest, X-ray Chest), and **Chatbot** (RAG + LangGraph medical assistant)—into a single application running on 8GB RAM laptops without internet. Key innovations: human-in-the-loop verification for uncertain predictions (35-45% confidence), WHO-aligned severity classification, classical CV quality checks, and clinical PDF/CSV report generation. Built for Nigerian hospitals with specialist shortages.

---

## For Abstract
- **Problem**: Specialist shortage (1 pathologist/1M pop), unreliable infrastructure, diagnostic delays
- **Solution**: Unified offline platform with 3 integrated modules
- **Methods**: YOLOv8n detection + UNet segmentation, INT8 quantization, ONNX Runtime, ChromaDB RAG, LangGraph orchestration
- **Results**: Malaria mAP@0.5 63.1%, parasitemia R²=0.87; MRI tumor Dice 0.81; CT nodule mAP 68.9%; X-ray pathology mAP 71.2%; all <500ms CPU inference, <6GB RAM
- **Impact**: Decision support for lab technicians/radiologists; air-gapped deployment; open-source (MIT)

---

## Key Technical Claims (Verifiable)
| Claim | Evidence Location |
|-------|-------------------|
| 4 blood diseases + 3 radiology modalities + RAG chatbot | `app/modules/` |
| CPU-only, <8GB RAM, offline | `requirements.txt`, `src/inference/model_loader.py` |
| Uncertainty flagging 35-45% | `app/modules/detection/malaria.py:33-34` |
| Human verification with real-time recalc | `app/streamlit_app.py:2430-2580` |
| Slide quality assessment (blur/exposure/contrast) | `src/utils/quality_check.py` |
| Clinical PDF reports with WHO severity | `src/utils/report_generator.py` |
| Session persistence (SQLite + JSON) | `src/utils/session_manager.py` |
| Lazy model loading + ONNX support | `src/inference/model_loader.py` |

---

## For Methods Section
**Architecture**: Streamlit frontend → Module router → Per-disease YOLOv8n detectors → Verification UI → Report generator
**Models**: YOLOv8n (quantized to ONNX INT8) for all detection; UNet for MRI segmentation
**Optimization**: Knowledge distillation (YOLOv8m→YOLOv8n), structured pruning, ONNX Runtime CPU, mmap weights
**Quality**: Laplacian variance (blur), mean intensity + saturation % (exposure), std dev (contrast)
**Verification**: Per-detection Accept/Reject buttons → immediate parasitemia/severity recomputation
**RAG**: all-MiniLM-L6-v2 embeddings → ChromaDB → top-5 retrieval → LangGraph intent classification → grounded generation → citation verification

---

## For Results Table
| Module | Task | Dataset | mAP@0.5 / Dice | Inference | Params |
|--------|------|---------|----------------|-----------|--------|
| Malaria | 5-class detection | BBBC041 | 63.1% | 140ms | 3.2M |
| Sickle Cell | 4-class detection | Custom | 71.2% | 130ms | 3.2M |
| ALL | 3-class detection | C-NMC | 68.4% | 120ms | 3.2M |
| Iron Deficiency | 4-class detection | Custom | 69.8% | 135ms | 3.2M |
| MRI | Tumor detection + seg | BraTS | 72.3% / 0.81 Dice | 320ms | 3.2M + 7.8M |
| CT | 4-class detection | LIDC | 68.9% | 280ms | 3.2M |
| X-ray | 4-class detection | CheXpert | 71.2% | 190ms | 3.2M |

---

## For Discussion Points
1. **Strengths**: Offline-first, unified platform, human-in-loop safety, WHO alignment, <6GB RAM
2. **Limitations**: Single-center malaria data (BBBC041), no prospective validation, CPU slower than GPU, limited rare classes
3. **Safety**: No standalone diagnosis; mandatory disclaimers; escalation for low confidence; audit trail
4. **Deployment**: Tested on 8GB RAM laptops (i7-1165G7, Ryzen 7 5800H); Windows/Linux; no internet needed

---

## For Reproducibility
```bash
# Environment
python 3.10+, torch 2.3.0+cpu, ultralytics 8.3.40
# Weights
models/detection/{malaria,sickle_cell,all,iron_deficiency}_yolov8n.pt
models/radiology/{mri,ct,xray}_yolov8n.pt
# Run
streamlit run app/streamlit_app.py
```

---

## Citation
```bibtex
@software{raphaid2026,
  title={RaphaID AI: Offline Multi-Disease Diagnostic Tool for African Healthcare},
  author={Akoleaje, Gabriel and Olajide, Treasure and Gbadegesin, Sodiq},
  year={2026},
  url={https://github.com/devions-forever/RaphaID_AI}
}
```

---

## Files for Reviewers
- **Full Notes**: `RESEARCH_PAPER_NOTES.md` (this directory)
- **Main App**: `app/streamlit_app.py`
- **Architecture Diagram**: See README.md
- **Dependencies**: `requirements.txt` (pinned)
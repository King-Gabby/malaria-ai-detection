# RaphaID AI - Project Specification

## Project Overview
**RaphaID AI** - Offline multi-disease diagnostic tool for lab technicians, scientists, and doctors in Nigerian/African hospitals with inadequate specialist personnel.

### Constraints
- Lightweight on-device software
- Must run on 8GB RAM constrained laptop
- Fully offline capability
- Models trained separately and optimized for lightweight deployment

---

## Architecture: 3 Main Parts

### 1. Detection
- Malaria
- Sickle Cell Disease
- ALL (Acute Lymphoblastic Leukemia)
- Iron Deficiency

### 2. Radiology
- MRI
- CT Scan
- X-ray

### 3. Chatbot
- RAG-powered medical assistant
- Orchestrated via LangGraph

---

## Technical Stack
- **Orchestrator**: LangGraph (coordinates models + RAG)
- **UI**: Streamlit + CSS + HTML
- **Models**: Separately trained, quantized/optimized for 8GB RAM
- **Session**: User info stored session-based (following PlasmoID pattern)
- **Analytics**: Metrics and analytics dashboard included

---

## UI/UX Specification

### Color Palette
| Role | Light Mode | Dark Mode |
|------|------------|-----------|
| Primary | `#0F766E` (Clinical Teal) | `#0F766E` |
| Accent/Success | `#10B981` (Emerald) | `#10B981` |
| Background | `#F9FAFB` | `#111827` |
| Text/Neutral | `#1F2937` | `#F9FAFB` |

### Layout Structure
```
┌─────────────────────────────────────┐
│  RaphaID Logo/Title                 │
├─────────────────────────────────────┤
│  [Detection ▼]  [Radiology ▼]  [Chatbot] │
├─────────────────────────────────────┤
│                                     │
│        Main Content Area            │
│                                     │
├─────────────────────────────────────┤
│  Session Info | Metrics | Analytics │
└─────────────────────────────────────┘
```

### Dropdown Menus

**Detection Dropdown** (clickable buttons):
- Malaria
- Sickle Cell Disease
- ALL
- Iron Deficiency

**Radiology Dropdown** (clickable buttons):
- MRI
- CT Scan
- X-ray

**Chatbot**: Clickable button (not dropdown)

### Splash Screen
- Refactor from PlasmoID to RaphaID branding
- Clinical Teal/Emerald theme
- Loading animation with progress indicator

---

## Workflow (Following PlasmoID Pattern)
1. Splash screen → Main dashboard
2. User selects module (Detection/Radiology/Chatbot)
3. Sub-selection for specific disease/modality
4. Upload/input medical data
5. Run inference (offline models)
6. Display results with confidence scores
7. Save to session history
8. Update metrics/analytics

---

## Model Optimization Strategy
- Quantization (INT8/INT4)
- Model pruning
- Knowledge distillation
- ONNX Runtime / TensorRT for inference
- Separate model files per disease/modality
- Lazy loading (load only when needed)

---

## Session Management
- In-memory session state (Streamlit session_state)
- Persistent local storage option (SQLite/JSON)
- User preferences, history, analytics cached locally
- No cloud dependency

---

## Next Steps
1. Refactor UI: PlasmoID → RaphaID (naming, colors, splash)
2. Implement dropdown navigation structure
3. Add session-based user info + metrics/analytics panels
4. Integrate LangGraph orchestrator skeleton
5. Set up model loading pipeline (lazy, quantized)
6. Build Detection module UI + inference pipeline
7. Build Radiology module UI + inference pipeline
8. Build Chatbot with RAG + LangGraph
9. End-to-end testing on 8GB RAM device

## current opencode steps
## Task 1 COMPLETE - Moving to Task 2
[x] Update color scheme: PlasmoID red/cyan → RaphaID Clinical Teal/Emerald
[x] Refactor splash screen with RaphaID branding
[x] Implement 3-module navigation (Detection, Radiology, Chatbot) with dropdowns
[x] Create Detection page with disease sub-menu (Malaria, Sickle Cell, ALL, Iron Deficiency)
[x] Create Radiology page with modality sub-menu (MRI, CT, X-ray)
[x] Create Chatbot page with RAG + LangGraph skeleton
[x] Add session info, metrics, analytics panels in footer
[x] Update all hardcoded PlasmoID references

## Task 2: Next Steps
[ ] Set up model loading pipeline (lazy, quantized)
[ ] Integrate LangGraph orchestrator skeleton
[ ] Build Detection module inference pipeline (Malaria)
[ ] Build Radiology module inference pipeline
[ ] Build Chatbot with RAG + LangGraph
[ ] End-to-end testing on 8GB RAM device
# Bari — Anemia Detection System

A multimodal machine learning system that predicts **anemia severity** from conjunctiva (inner eyelid) images and patient clinical data. Combines a fine-tuned MobileNetV2 visual model with tabular classifiers in a TFLite fusion network, served via a FastAPI REST API.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Models](#models)
5. [Installation](#installation)
6. [Running the API](#running-the-api)
7. [API Endpoints](#api-endpoints)
8. [Input / Output Reference](#input--output-reference)
9. [Example Requests](#example-requests)
10. [Generating Model Files from the Notebook](#generating-model-files-from-the-notebook)
11. [Dataset Citation](#dataset-citation)
12. [Clinical Disclaimer](#clinical-disclaimer)

---

## Project Overview

The system classifies anemia severity into four WHO-aligned categories:

| Class | Hemoglobin (g/dL) — Female | Hemoglobin (g/dL) — Male |
|-------|---------------------------|--------------------------|
| **Normal** | ≥ 12.0 | ≥ 13.5 |
| **Mild** | 10.0 – 11.9 | 11.0 – 13.4 |
| **Moderate** | 7.0 – 9.9 | 7.0 – 9.9 |
| **Severe** | < 7.0 | < 7.0 |

Prediction can be done from:
- **Tabular data only** (age, gender, optional HB level)
- **Image only** (conjunctiva photograph)
- **Multimodal fusion** (image + tabular, best accuracy)

All predictions include nutritional guidance, recommended foods, and a referral action.

---

## System Architecture

```
INPUT PIPELINE
├── Conjunctiva Image (160×160 RGB)
│   └── MobileNetV2 (TFLite) → Visual embeddings
│
├── Tabular Features
│   ├── With HB: [HB_LEVEL, Age(Months), GENDER]  → Random Forest + Scaler
│   └── No HB:  [Age(Months), GENDER]              → Random Forest + Scaler
│
└── FUSION (TFLite)
    ├── Image embeddings + Tabular probabilities
    └── Dense layers → Softmax → {Normal, Mild, Moderate, Severe}

OUTPUT
├── prediction          — severity class string
├── confidence          — top-class probability (0–1)
├── class_probabilities — {Normal, Mild, Moderate, Severe} dict
├── nutrition           — dietary advice paragraph
├── recommended_foods   — list of food suggestions
└── referral_action     — clinical follow-up recommendation
```

---

## Directory Structure

```
Bari/
├── README.md                          # This file
│
├── app/                               # FastAPI application
│   ├── main.py                        # Entry point — all 5 endpoints
│   ├── requirements.txt               # API dependencies
│   ├── schemas/
│   │   ├── request.py                 # Pydantic request models
│   │   └── response.py                # Pydantic response models
│   ├── services/
│   │   ├── inference.py               # Model inference logic
│   │   ├── nutrition.py               # Guidance generation
│   │   └── preprocessing.py           # Image & tabular preprocessing
│   └── utils/
│       ├── image_utils.py
│       └── tabular_utils.py
│
├── models/
│   └── saved_models/
│       ├── visual_model.tflite        # MobileNetV2 quantized (image-only)
│       ├── multimodal_model.tflite    # Fusion model (image + HB + age + gender)
│       └── multimodal_no_hb_model.tflite  # Fusion model (image + age + gender)
│
├── Notebook/
│   ├── Bari.ipynb                     # Full training notebook
│   ├── models/
│   │   ├── tabular_with_hb.pkl        # RF + scaler bundle (HB, Age, Gender)
│   │   └── tabular_no_hb.pkl          # RF + scaler bundle (Age, Gender)
│   └── results/                       # Training plots, CSVs, confusion matrices
│
├── data/
│   ├── Images/
│   │   ├── Anemic/                    # Conjunctiva images — anemic patients
│   │   └── Non-anemic/                # Conjunctiva images — healthy individuals
│   └── Tabular/
│       └── anemia.csv                 # Clinical data (Age, Gender, HB, Severity)
│
├── mobile/                            # React Native mobile client
└── nutrition/                         # Nutritional guidance data
```

---

## Models

### Tabular Classifiers (trained in notebook, used by API)

| Model | File | Features | Notes |
|-------|------|----------|-------|
| Random Forest + Scaler | `Notebook/models/tabular_with_hb.pkl` | HB_LEVEL, Age(Months), GENDER | Primary model |
| Random Forest + Scaler | `Notebook/models/tabular_no_hb.pkl` | Age(Months), GENDER | Fallback when HB unavailable |
| Logistic Regression | `Notebook/results/Tuned_Logistic_Regression.pkl` | Same splits | Reference comparison |
| XGBoost | `Notebook/results/Tuned_XGBoost.pkl` | Same splits | Reference comparison |

Both `.pkl` files are joblib bundles: `{"model": sklearn_model, "scaler": StandardScaler}`.

### TFLite Models

| File | Input | Use case |
|------|-------|----------|
| `visual_model.tflite` | `[1, 160, 160, 3]` float32 | Image-only prediction |
| `multimodal_model.tflite` | image `[1,160,160,3]` + tabular `[1,3]` | Fusion with HB |
| `multimodal_no_hb_model.tflite` | image `[1,160,160,3]` + tabular `[1,2]` | Fusion without HB |

---

## Installation

### Prerequisites

- Python 3.9 – 3.11 (TensorFlow required for image/multimodal endpoints)
- Python 3.12+ works for tabular-only mode (TF skipped automatically at startup)
- pip ≥ 23

### Steps

```bash
# 1. Clone / navigate to the project root
cd Bari

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install API dependencies
pip install -r app/requirements.txt
```

**Key dependencies:** fastapi, uvicorn, scikit-learn, joblib, numpy, Pillow, tensorflow (optional), shap (optional)

---

## Running the API

Run all commands from the **project root** (`Bari/`), not from inside `app/`.

### Development (auto-reload)

```bash
uvicorn app.main:app --reload --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Specific host

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### As a Python module

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Background process (Unix)

```bash
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

After starting, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/

### Environment Variables (optional model path overrides)

| Variable | Default path | Description |
|----------|-------------|-------------|
| `TAB_WH_PATH` | `Notebook/models/tabular_with_hb.pkl` | RF model with HB |
| `TAB_NH_PATH` | `Notebook/models/tabular_no_hb.pkl` | RF model without HB |
| `VIS_PATH` | `models/saved_models/visual_model.tflite` | Visual TFLite |
| `MM_WH_PATH` | `models/saved_models/multimodal_model.tflite` | Fusion with HB TFLite |
| `MM_NH_PATH` | `models/saved_models/multimodal_no_hb_model.tflite` | Fusion without HB TFLite |

Set via shell, `.env` file (with `python-dotenv`), or inline:

```bash
# Linux / macOS
export TAB_WH_PATH=/custom/path/tabular_with_hb.pkl

# Windows
set TAB_WH_PATH=C:\custom\path\tabular_with_hb.pkl
```

---

## API Endpoints

### GET `/` — Health Check

Returns API version and which models are loaded.

**Response:**
```json
{
  "status": "API running",
  "version": "2.0.0",
  "models_loaded": {
    "tabular_with_hb": true,
    "tabular_no_hb": true,
    "visual_tflite": false,
    "multimodal_with_hb": false,
    "multimodal_no_hb": false
  }
}
```

---

### POST `/predict/tabular` — Tabular Prediction

Predict anemia severity from clinical features only (no image required).

**Request body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `age` | float | Yes | Age in **months** (0 – 1200) |
| `gender` | int | Yes | `0` = Male, `1` = Female |
| `hb_level` | float \| null | No | Hemoglobin in g/dL (0 – 25). Omit or `null` to use no-HB model |

**Model routing:**
- `hb_level` provided → `tabular_with_hb.pkl` (features: HB_LEVEL, Age, GENDER)
- `hb_level` absent → `tabular_no_hb.pkl` (features: Age, GENDER)

---

### POST `/predict/image` — Image Prediction

Predict anemia from a conjunctiva photograph using the TFLite MobileNetV2 model.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Conjunctiva image — JPEG or PNG, any resolution (resized to 160×160 internally) |

Requires `visual_model.tflite` to be present.

---

### POST `/predict/multimodal` — Multimodal Fusion Prediction

Best-accuracy endpoint. Combines the conjunctiva image with patient clinical data.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Conjunctiva image (JPEG / PNG) |
| `age` | float | Yes | Age in months |
| `gender` | int | Yes | `0` = Male, `1` = Female |
| `hb_level` | float | No | Hemoglobin g/dL — determines which fusion model is used |

**Model routing:**
- `hb_level` present → `multimodal_model.tflite` (tabular branch: 3 features)
- `hb_level` absent → `multimodal_no_hb_model.tflite` (tabular branch: 2 features)

---

### POST `/explain/tabular` — SHAP Feature Explanation

Returns SHAP feature importance scores for a tabular prediction. Requires `shap` installed.

**Request body (JSON):** Same fields as `/predict/tabular`.

**Response:** `ExplainResponse` — includes `top_features` dict (feature → mean |SHAP|), prediction, confidence, nutrition advice, and a note.

---

## Input / Output Reference

### All prediction endpoints share this response schema

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"Normal"` / `"Mild"` / `"Moderate"` / `"Severe"` |
| `confidence` | float | Top class probability (0.0 – 1.0) |
| `class_probabilities` | object | `{Normal: f, Mild: f, Moderate: f, Severe: f}` |
| `nutrition` | string | Dietary advice paragraph |
| `recommended_foods` | list[string] | Specific food items |
| `referral_action` | string | Clinical follow-up recommendation |

### Feature vectors per endpoint variant

| Endpoint | Variant | Feature vector sent to model |
|----------|---------|------------------------------|
| `/predict/tabular` | with HB | `[HB_LEVEL, Age(months), GENDER]` — scaled |
| `/predict/tabular` | no HB | `[Age(months), GENDER]` — scaled |
| `/predict/image` | — | Image resized to `[1, 160, 160, 3]` float32, `/255` |
| `/predict/multimodal` | with HB | Image `[1,160,160,3]` + tabular `[1,3]` |
| `/predict/multimodal` | no HB | Image `[1,160,160,3]` + tabular `[1,2]` |
| `/explain/tabular` | with HB | Same as tabular with HB |
| `/explain/tabular` | no HB | Same as tabular no HB |

---

## Example Requests

### Tabular — with HB (curl)

```bash
curl -X POST http://localhost:8000/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{"age": 312, "gender": 1, "hb_level": 9.5}'
```

### Tabular — without HB (Python)

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict/tabular",
    json={"age": 312, "gender": 1}
)
print(resp.json())
```

### Image prediction (curl)

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "file=@conjunctiva.jpg"
```

### Multimodal fusion (Python)

```python
import requests

with open("conjunctiva.jpg", "rb") as img:
    resp = requests.post(
        "http://localhost:8000/predict/multimodal",
        files={"file": img},
        data={"age": 312, "gender": 1, "hb_level": 9.5}
    )
print(resp.json())
```

### SHAP explanation (curl)

```bash
curl -X POST http://localhost:8000/explain/tabular \
  -H "Content-Type: application/json" \
  -d '{"age": 312, "gender": 1, "hb_level": 9.5}'
```

**Example response:**
```json
{
  "prediction": "Moderate",
  "confidence": 0.8241,
  "class_probabilities": {
    "Normal": 0.0312,
    "Mild": 0.1104,
    "Moderate": 0.8241,
    "Severe": 0.0343
  },
  "nutrition": "Increase iron-rich foods. Consider iron supplementation under medical supervision.",
  "recommended_foods": ["spinach", "lentils", "lean red meat", "fortified cereals"],
  "referral_action": "Refer to physician for further evaluation within 1–2 weeks."
}
```

### React Native integration

```javascript
const formData = new FormData();
formData.append('file', { uri: imageUri, type: 'image/jpeg', name: 'eye.jpg' });
formData.append('age', '312');
formData.append('gender', '1');
formData.append('hb_level', '9.5');

const response = await fetch('http://<server>/predict/multimodal', {
  method: 'POST',
  body: formData,
});
const result = await response.json();
```

---

## Error Codes

| Status | Meaning | Example cause |
|--------|---------|---------------|
| 400 | Bad Request | Preprocessing failed (e.g. non-numeric age) |
| 422 | Unprocessable Entity | Missing required field, unsupported image type |
| 501 | Not Implemented | `shap` not installed, `/explain/tabular` called |
| 503 | Service Unavailable | Required model file not found at startup |
| 500 | Internal Server Error | Unexpected inference failure |

---

## Generating Model Files from the Notebook

Run `Notebook/Bari.ipynb` sequentially in Google Colab or locally (Python 3.9–3.11 with TensorFlow).

| Notebook section | Output file | Used by endpoint |
|-----------------|-------------|-----------------|
| Part 5 — Random Forest (with HB) | `Notebook/models/tabular_with_hb.pkl` | `/predict/tabular`, `/predict/multimodal`, `/explain/tabular` |
| Part 5 — Random Forest (no HB) | `Notebook/models/tabular_no_hb.pkl` | Same, fallback variant |
| Part 7 — Visual TFLite export | `models/saved_models/visual_model.tflite` | `/predict/image` |
| Part 8 — Fusion TFLite export | `models/saved_models/multimodal_model.tflite` | `/predict/multimodal` (with HB) |
| Part 8 — Fusion TFLite export | `models/saved_models/multimodal_no_hb_model.tflite` | `/predict/multimodal` (no HB) |

---

## Dataset Citation

Asare, Justice Williams; APPIAHENE, PETER; DONKOH, EMMANUEL (2023),
"CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana",
Mendeley Data, V1, doi: [10.17632/m53vz6b7fx.1](https://doi.org/10.17632/m53vz6b7fx.1)

---

## Clinical Disclaimer

**This system is for educational and research purposes only.**

- NOT a diagnostic tool — use only as a preliminary screening aid
- Always consult qualified healthcare professionals for diagnosis and treatment
- Developers assume no responsibility for medical decisions based on this system
- Must be used alongside, never instead of, proper clinical examination

---

## License

Provided as-is for educational and research purposes.

---

## Contributors

- **Project**: Bari Anemia Detection System
- **Institution**: Capstone Project
- **Year**: 2026

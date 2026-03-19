# Anemia Screening REST API

FastAPI application that predicts anemia severity from patient tabular data, conjunctiva images, or both combined. Backed by trained scikit-learn models and quantized TFLite neural networks, with SHAP explainability and rule-based nutritional guidance.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Model Files](#model-files)
5. [Running the API](#running-the-api)
6. [Environment Variables](#environment-variables)
7. [Endpoints](#endpoints)
   - [GET / — Health Check](#get----health-check)
   - [POST /predict/tabular — Tabular Prediction](#post-predicttabular--tabular-prediction)
   - [POST /predict/image — Image Prediction](#post-predictimage--image-prediction)
   - [POST /predict/multimodal — Multimodal Prediction](#post-predictmultimodal--multimodal-prediction)
   - [POST /explain/tabular — SHAP Explanation](#post-explaintabular--shap-explanation)
8. [Input / Output Reference](#input--output-reference)
9. [Class Mapping](#class-mapping)
10. [Error Codes](#error-codes)
11. [React Native Integration](#react-native-integration)
12. [Generating Model Files from the Notebook](#generating-model-files-from-the-notebook)

---

## Project Structure

```
Bari/
├── Notebook/
│   ├── Bari.ipynb              ← Training notebook (generates all model files)
│   └── models/
│       ├── tabular_with_hb.pkl ← RandomForest + scaler bundle (with HB_LEVEL)
│       └── tabular_no_hb.pkl   ← RandomForest + scaler bundle (no HB_LEVEL)
├── models/
│   └── saved_models/
│       ├── visual_model.tflite           ← Quantized MobileNetV2
│       ├── multimodal_model.tflite       ← Fusion model (image + 3 tabular features)
│       └── multimodal_no_hb_model.tflite ← Fusion model (image + 2 tabular features)
└── app/
    ├── main.py                 ← FastAPI app, lifespan loading, all endpoints
    ├── requirements.txt
    ├── schemas/
    │   ├── request.py          ← Pydantic input models
    │   └── response.py         ← Pydantic output models
    ├── services/
    │   ├── inference.py        ← Stateless model-call functions
    │   ├── preprocessing.py    ← Preprocessing facade
    │   └── nutrition.py        ← Rule-based nutritional guidance engine
    └── utils/
        ├── image_utils.py      ← PIL load / resize / normalise helpers
        └── tabular_utils.py    ← Feature vector builder + scaler wrapper
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 – 3.11 (TFLite requires ≤ 3.11) |
| pip | ≥ 22 |
| TensorFlow | ≥ 2.10 (for image / multimodal endpoints) |

> **Note:** The tabular-only endpoints (`/predict/tabular`, `/explain/tabular`) work without TensorFlow. Image and multimodal endpoints require TensorFlow to be installed.

---

## Installation

```bash
# 1. Clone / navigate to the project root
cd Bari

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r app/requirements.txt
```

To install without TensorFlow (tabular endpoints only):

```bash
pip install fastapi "uvicorn[standard]" python-multipart pydantic \
            scikit-learn numpy joblib Pillow python-dotenv
```

---

## Model Files

The API looks for model files in these default locations (relative to the project root):

| Key | Default Path | Required For |
|-----|-------------|--------------|
| `tabular_with_hb` | `Notebook/models/tabular_with_hb.pkl` | `/predict/tabular` (with HB), `/explain/tabular` |
| `tabular_no_hb` | `Notebook/models/tabular_no_hb.pkl` | `/predict/tabular` (no HB), `/explain/tabular` |
| `visual_model.tflite` | `models/saved_models/visual_model.tflite` | `/predict/image` |
| `multimodal_model.tflite` | `models/saved_models/multimodal_model.tflite` | `/predict/multimodal` (with HB) |
| `multimodal_no_hb_model.tflite` | `models/saved_models/multimodal_no_hb_model.tflite` | `/predict/multimodal` (no HB) |

Each `.pkl` file is a `joblib` bundle with the structure:
```python
{"model": <fitted sklearn model>, "scaler": <fitted StandardScaler>}
```

The API starts successfully even if some files are missing — missing models simply make their corresponding endpoints return `503 Service Unavailable`.

---

## Running the API

All commands are run from the **project root** (`Bari/`), not from inside `app/`.

### Development (auto-reload on file changes)

```bash
uvicorn app.main:app --reload --port 8000
```

### Production (multi-worker)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Specific host / port

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

### Using Python directly

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Background process (Unix)

```bash
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

Once running, the interactive documentation is available at:

| Interface | URL |
|-----------|-----|
| Swagger UI (interactive) | http://127.0.0.1:8000/docs |
| ReDoc (readable) | http://127.0.0.1:8000/redoc |
| OpenAPI JSON schema | http://127.0.0.1:8000/openapi.json |

---

## Environment Variables

Override any default model path without changing code:

| Variable | Overrides | Example |
|----------|-----------|---------|
| `TAB_WH_PATH` | `tabular_with_hb.pkl` path | `/data/models/tab_wh.pkl` |
| `TAB_NH_PATH` | `tabular_no_hb.pkl` path | `/data/models/tab_nh.pkl` |
| `VIS_PATH` | `visual_model.tflite` path | `/data/models/visual.tflite` |
| `MM_WH_PATH` | `multimodal_model.tflite` path | `/data/models/mm_wh.tflite` |
| `MM_NH_PATH` | `multimodal_no_hb_model.tflite` path | `/data/models/mm_nh.tflite` |

**Linux / macOS:**
```bash
export TAB_WH_PATH=/custom/path/tabular_with_hb.pkl
export VIS_PATH=/custom/path/visual_model.tflite
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:TAB_WH_PATH = "C:\custom\path\tabular_with_hb.pkl"
$env:VIS_PATH    = "C:\custom\path\visual_model.tflite"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Using a `.env` file** (place in `app/`):
```env
TAB_WH_PATH=C:/custom/path/tabular_with_hb.pkl
VIS_PATH=C:/custom/path/visual_model.tflite
```

---

## Endpoints

### `GET /` — Health Check

Returns API version and which models are currently loaded.

```bash
curl http://127.0.0.1:8000/
```

**Response `200 OK`**
```json
{
  "status": "API running",
  "version": "2.0.0",
  "models_loaded": {
    "tabular_with_hb":    true,
    "tabular_no_hb":      true,
    "visual_tflite":      true,
    "multimodal_with_hb": true,
    "multimodal_no_hb":   true
  }
}
```

---

### `POST /predict/tabular` — Tabular Prediction

Predict anemia severity from patient demographics and optionally haemoglobin level.

- **With `hb_level`** → routes to the `tabular_with_hb` RandomForest (features: `[HB_LEVEL, Age(Months), Gender_F]`)
- **Without `hb_level`** → routes to the `tabular_no_hb` RandomForest (features: `[Age(Months), Gender_F]`)

#### Request Body (`application/json`)

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `age` | float | Yes | 0 – 1200 | Patient age **in months** |
| `gender` | int | Yes | 0 or 1 | `0` = Male, `1` = Female |
| `hb_level` | float | No | 0.0 – 25.0 | Haemoglobin level in g/dL |

**With HB_LEVEL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"age": 24, "gender": 1, "hb_level": 9.5}'
```

**Without HB_LEVEL (demographic baseline):**
```bash
curl -X POST "http://127.0.0.1:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"age": 36, "gender": 0}'
```

**Python (requests):**
```python
import requests

# With HB
r = requests.post("http://127.0.0.1:8000/predict/tabular",
                  json={"age": 24, "gender": 1, "hb_level": 9.5})
print(r.json())

# Without HB
r = requests.post("http://127.0.0.1:8000/predict/tabular",
                  json={"age": 36, "gender": 0})
print(r.json())
```

#### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Predicted class: `"Non-Anemic"`, `"Mild"`, `"Moderate"`, or `"Severe"` |
| `confidence` | float | Probability of the predicted class (0.0 – 1.0) |
| `class_probabilities` | object | Probability for **each** of the 4 classes |
| `nutrition` | string | Short nutritional advice tailored to the predicted class |
| `recommended_foods` | array[string] | List of foods recommended for the predicted class |
| `referral_action` | string | Clinical referral instruction |

```json
{
  "prediction": "Moderate",
  "confidence": 0.82,
  "class_probabilities": {
    "Non-Anemic": 0.05,
    "Mild":       0.08,
    "Moderate":   0.82,
    "Severe":     0.05
  },
  "nutrition": "Structured dietary plan: iron, folate, and B12 required.",
  "recommended_foods": ["Liver", "Red meat", "Dark leafy greens", "Fortified cereals"],
  "referral_action": "Medical consultation within 2 weeks. Request CBC + serum ferritin."
}
```

---

### `POST /predict/image` — Image Prediction

Predict anemia severity from a conjunctiva eye image using a quantized MobileNetV2 TFLite model.

**Accepted image formats:** `image/jpeg`, `image/jpg`, `image/png`, `image/webp`

#### Request (`multipart/form-data`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Conjunctiva eye image (JPEG or PNG) |

```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
  -F "file=@/path/to/conjunctiva.jpg"
```

**Python (requests):**
```python
import requests

with open("conjunctiva.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict/image",
                      files={"file": ("conjunctiva.jpg", f, "image/jpeg")})
print(r.json())
```

#### Response `200 OK`

Same shape as `/predict/tabular`. `nutrition`, `recommended_foods`, and `referral_action` are derived from the predicted class; no age/gender context is available for image-only predictions.

```json
{
  "prediction": "Mild",
  "confidence": 0.74,
  "class_probabilities": {
    "Non-Anemic": 0.12,
    "Mild":       0.74,
    "Moderate":   0.10,
    "Severe":     0.04
  },
  "nutrition": "Increase dietary iron and vitamin C intake.",
  "recommended_foods": ["Spinach", "Lentils", "Kidney beans", "Bell peppers"],
  "referral_action": "Schedule a physician visit within 4 weeks."
}
```

**Image preprocessing pipeline** (applied internally):
1. Decode JPEG / PNG → RGB
2. Resize to 180 × 180
3. Centre-crop to 160 × 160
4. Normalise pixel values to [0, 1]

---

### `POST /predict/multimodal` — Multimodal Prediction

Fuse conjunctiva image with tabular patient data using a TFLite fusion model (MobileNetV2 image branch + dense tabular branch).

- **With `hb_level`** → `multimodal_model.tflite` (tabular features: `[HB_LEVEL, Age(Months), Gender_F]`)
- **Without `hb_level`** → `multimodal_no_hb_model.tflite` (tabular features: `[Age(Months), Gender_F]`)

#### Request (`multipart/form-data`)

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `file` | file | Yes | JPEG / PNG / WebP | Conjunctiva eye image |
| `age` | float | Yes | 0 – 1200 | Patient age in months |
| `gender` | int | Yes | 0 or 1 | `0` = Male, `1` = Female |
| `hb_level` | float | No | 0.0 – 25.0 | Haemoglobin level g/dL |

**With HB level:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/multimodal" \
  -F "file=@/path/to/conjunctiva.jpg" \
  -F "age=24" \
  -F "gender=1" \
  -F "hb_level=8.0"
```

**Without HB level:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/multimodal" \
  -F "file=@/path/to/conjunctiva.jpg" \
  -F "age=36" \
  -F "gender=0"
```

**Python (requests):**
```python
import requests

with open("conjunctiva.jpg", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/predict/multimodal",
        files={"file": ("conjunctiva.jpg", f, "image/jpeg")},
        data={"age": 24, "gender": 1, "hb_level": 8.0},
    )
print(r.json())
```

#### Response `200 OK`

Same shape as `/predict/tabular`. Nutritional guidance is personalised using both the predicted class and the provided age/gender.

---

### `POST /explain/tabular` — SHAP Explanation

Returns SHAP (SHapley Additive exPlanations) feature importance scores for a tabular prediction.

> Requires `shap` package: `pip install shap`

#### Request Body (`application/json`)

Same fields as `/predict/tabular` — `age`, `gender`, optional `hb_level`.

```bash
curl -X POST "http://127.0.0.1:8000/explain/tabular" \
  -H "Content-Type: application/json" \
  -d '{"age": 24, "gender": 1, "hb_level": 9.5}'
```

**Python (requests):**
```python
import requests

r = requests.post("http://127.0.0.1:8000/explain/tabular",
                  json={"age": 24, "gender": 1, "hb_level": 9.5})
print(r.json())
```

#### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Predicted severity class |
| `confidence` | float | Model confidence (0.0 – 1.0) |
| `top_features` | object | Feature name → mean absolute SHAP value (higher = more influential) |
| `nutrition` | string | Short nutritional advice |
| `note` | string | Explanation of SHAP interpretation |

```json
{
  "prediction": "Moderate",
  "confidence": 0.82,
  "top_features": {
    "HB_LEVEL":    0.412,
    "Age(Months)": 0.087,
    "Gender_F":    0.031
  },
  "nutrition": "Iron, folate, and B12 required.",
  "note": "SHAP values show mean |contribution| per feature across all classes. Higher = more influential."
}
```

**Feature interpretation:**
- `HB_LEVEL` — haemoglobin level (present only when `hb_level` was provided)
- `Age(Months)` — patient age
- `Gender_F` — gender binary flag (1.0 = Female, 0.0 = Male)

---

## Input / Output Reference

### Shared Input Fields

| Field | Type | Range | Encoding | Used By |
|-------|------|-------|----------|---------|
| `age` | float | 0 – 1200 months | Raw numeric | All tabular endpoints |
| `gender` | int | 0 or 1 | `0` = Male, `1` = Female | All tabular endpoints |
| `hb_level` | float (optional) | 0.0 – 25.0 g/dL | Raw numeric | All tabular endpoints |
| `file` | image file | — | JPEG / PNG / WebP | Image, multimodal endpoints |

> **Age is in months, not years.** Examples: 6-month-old infant → `6`, 2-year-old → `24`, 10-year-old → `120`, 25-year-old adult → `300`.

### Shared Output Fields

| Field | Type | Always Present | Description |
|-------|------|---------------|-------------|
| `prediction` | string | Yes | One of: `"Non-Anemic"`, `"Mild"`, `"Moderate"`, `"Severe"` |
| `confidence` | float | Yes | Softmax probability of the top class (0.0 – 1.0) |
| `class_probabilities` | object | Yes | `{"Non-Anemic": p0, "Mild": p1, "Moderate": p2, "Severe": p3}` — all four probabilities always included, sum to ~1.0 |
| `nutrition` | string | Yes | Dietary advice sentence |
| `recommended_foods` | array[string] | Yes | 4 – 8 food items |
| `referral_action` | string | Yes | Clinical follow-up instruction |

### Feature Vectors Sent to Models

| Endpoint variant | Feature vector | Scaler applied |
|-----------------|----------------|----------------|
| `/predict/tabular` with `hb_level` | `[HB_LEVEL, Age(Months), Gender_F]` | `scaler_wh` (StandardScaler) |
| `/predict/tabular` without `hb_level` | `[Age(Months), Gender_F]` | `scaler_nh` (StandardScaler) |
| `/predict/image` | `(1, 160, 160, 3)` float32 normalised image | None |
| `/predict/multimodal` with `hb_level` | image + `[HB_LEVEL, Age(Months), Gender_F]` | `scaler_wh` on tabular branch |
| `/predict/multimodal` without `hb_level` | image + `[Age(Months), Gender_F]` | `scaler_nh` on tabular branch |

---

## Class Mapping

| Index | Label | HB_LEVEL Range | Clinical Meaning |
|-------|-------|----------------|-----------------|
| 0 | Non-Anemic | ≥ 12 g/dL (Female) / ≥ 13 g/dL (Male) | Normal haemoglobin |
| 1 | Mild | 10.0 – 11.9 g/dL | Mild anaemia; dietary intervention recommended |
| 2 | Moderate | 7.0 – 9.9 g/dL | Moderate anaemia; medical review required |
| 3 | Severe | < 7.0 g/dL | Severe anaemia; urgent medical attention |

---

## Error Codes

| HTTP Code | Condition | Example |
|-----------|-----------|---------|
| `400 Bad Request` | Invalid or missing input field | `age` out of range, `gender` not 0/1 |
| `422 Unprocessable Entity` | Corrupt image, empty file, unsupported format | Non-image file uploaded |
| `422 Unprocessable Entity` | Pydantic validation failure | Negative `hb_level` |
| `501 Not Implemented` | `shap` package not installed | Call to `/explain/tabular` without shap |
| `503 Service Unavailable` | Required model file not loaded | `.pkl` or `.tflite` file missing at startup |
| `500 Internal Server Error` | Unexpected inference failure | Model I/O shape mismatch |

---

## React Native Integration

```javascript
// POST /predict/multimodal from a React Native app
const predictAnemia = async (imageUri, age, gender, hbLevel = null) => {
  const formData = new FormData();

  formData.append("file", {
    uri: imageUri,
    type: "image/jpeg",
    name: "conjunctiva.jpg",
  });
  formData.append("age", String(age));
  formData.append("gender", String(gender));
  if (hbLevel !== null) {
    formData.append("hb_level", String(hbLevel));
  }

  const response = await fetch("http://YOUR_SERVER_IP:8000/predict/multimodal", {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail);
  }

  return await response.json();
  // Returns: { prediction, confidence, class_probabilities,
  //            nutrition, recommended_foods, referral_action }
};

// Usage
const result = await predictAnemia(imageUri, 300, 1, 9.5);
console.log(result.prediction);        // "Moderate"
console.log(result.confidence);        // 0.82
console.log(result.recommended_foods); // ["Liver", ...]
console.log(result.referral_action);   // "Medical consultation..."
```

---

## Generating Model Files from the Notebook

The API relies on model files produced by running `Notebook/Bari.ipynb`. Follow these steps:

### 1. Install notebook dependencies

```bash
pip install tensorflow scikit-learn xgboost shap pandas numpy matplotlib seaborn pillow joblib opencv-python
```

> TensorFlow requires Python 3.9 – 3.11.

### 2. Set the base path

In `Notebook/Bari.ipynb`, cell 3 (config), confirm:
```python
BASE_PATH = "c:/Users/USER/Capstone/Bari"   # adjust to your machine
```

### 3. Run all cells in order

**Kernel > Restart & Run All**

The notebook will produce:

| Output file | Cell | Used by |
|-------------|------|---------|
| `Notebook/models/tabular_with_hb.pkl` | Save sklearn models | `/predict/tabular`, `/explain/tabular` |
| `Notebook/models/tabular_no_hb.pkl` | Save sklearn models | `/predict/tabular`, `/explain/tabular` |
| `Notebook/models/visual_model.keras` | Visual training | TFLite converter |
| `Notebook/models/multimodal_model.keras` | Multimodal training | TFLite converter |
| `models/saved_models/visual_model.tflite` | TFLite conversion | `/predict/image` |
| `models/saved_models/multimodal_model.tflite` | TFLite conversion | `/predict/multimodal` |
| `models/saved_models/multimodal_no_hb_model.tflite` | TFLite conversion | `/predict/multimodal` |

### 4. Restart the API

```bash
uvicorn app.main:app --reload --port 8000
```

Confirm all models are loaded:
```bash
curl http://127.0.0.1:8000/
# all "models_loaded" values should be true
```

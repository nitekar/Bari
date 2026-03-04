
# 🩺 Anemia Detection System - Bari Project

## 📋 Project Overview

The **Bari Anemia Detection System** is a machine learning-powered application that predicts the severity of anemia by combining **visual analysis of the inner eyelid** with **clinical tabular data**. The system leverages a **fusion neural network** that integrates:

- **Visual Analysis**: Fine-tuned MobileNetV2 model analyzing eyelid images
- **Clinical Data**: Random Forest model processing age, gender, and hemoglobin levels
- **Explainability**: Grad-CAM visualization showing which regions of the eyelid influence predictions
- **Clinical Guidance**: Automated nutritional and lifestyle recommendations based on WHO guidelines

This system aims to provide a **non-invasive, accessible preliminary assessment** of anemia severity, potentially supporting healthcare professionals in early detection and intervention.

---

## 🏗️ System Architecture

```
INPUT PIPELINE:
├── Eyelid Image
│   ├── Image Preprocessing (224×224)
│   ├── MobileNetV2 Feature Extraction
│   └── Image Embeddings
│
├── Patient Tabular Data (Age, Gender, Hemoglobin)
│   ├── Random Forest Model
│   └── Probability Predictions
│
║
╚══> FUSION LAYER
     ├── Concatenate Image Embeddings + Tabular Probabilities
     ├── Fusion Neural Network
     └── Final Classification
          │
          ├─ Predicted Class (Normal, Mild, Moderate, Severe)
          ├─ Class Probabilities
          ├─ Grad-CAM Heatmap (visual explanation)
          ├─ Nutritional Recommendations
          └─ Clinical Advice
```

---

## ✨ Key Features

- **Multi-modal Prediction**: Combines image and tabular data for robust predictions
- **Interpretable Results**: Grad-CAM visualization shows which areas of the eyelid contribute to predictions
- **Clinical Recommendations**: Automatic nutritional and lifestyle advice based on prediction severity
- **REST API**: FastAPI-based endpoint for easy integration with other systems
- **Scalable**: Pre-trained MobileNetV2 backbone enables efficient inference

---

## 📊 Model Findings & Results

### Visual Model (MobileNetV2)
- **Architecture**: Pre-trained MobileNetV2 fine-tuned on eyelid imagery
- **Input Size**: 224×224 RGB images
- **Purpose**: Extract discriminative visual features from eyelid pallor patterns
- **Performance**: Effective at capturing subtle color and texture variations indicative of hemoglobin levels

### Tabular Model (Random Forest)
- **Features**: Age, Gender, Hemoglobin Level (Hb)
- **Purpose**: Predict anemia probability from clinical parameters
- **Advantage**: Handles non-linear relationships between clinical features and anemia severity
- **Models Trained**:
  - LogisticRegression_model.pkl
  - RandomForest_model.pkl (Primary)
  - XGBoost_model.pkl

### Fusion Model
- **Architecture**: Neural network combining image embeddings and tabular probabilities
- **Output Classes**: 
  - **Normal**: Hb ≥ 12.0 g/dL (female) / ≥ 13.5 g/dL (male)
  - **Mild**: Hb 10.0-11.9 / 11.0-13.4 g/dL
  - **Moderate**: Hb 7.0-9.9 g/dL
  - **Severe**: Hb < 7.0 g/dL

---

## 📊 Model Performance Metrics

### Tabular Models Performance

| Model | Accuracy | Precision | Recall | F1-Score | F2-Score | AUC |
|-------|----------|-----------|--------|----------|----------|-----|
| **LogisticRegression** | 98.59% | 97.70% | 100% | 98.84% | 99.53% | 1.00 |
| **RandomForest** | **100%** | **100%** | **100%** | **100%** | **100%** | **1.00** |
| **XGBoost** | **100%** | **100%** | **100%** | **100%** | **100%** | **1.00** |

**Key Insights:**
- ✅ **RandomForest** and **XGBoost** achieve perfect classification on tabular features
- ✅ Strong recall (100%) indicates no missed positive cases (anemia detection)
- ✅ High precision demonstrates minimal false positives
- ✅ Perfect AUC indicates excellent class separation

### Model Selection
- **Primary Tabular Model**: RandomForest (perfect performance, interpretable feature importance)
- **Fallback Models**: XGBoost (alternative), LogisticRegression (lightweight inference)

### Visual Model Performance
- **MobileNetV2 Fine-tuning**: Optimized for eyelid imagery classification
- **Feature Extraction**: 1280-dimensional embeddings capture discriminative visual patterns
- **Validation**: Model validated on held-out eyelid image dataset

### Fusion Model Integration
- **End-to-End Accuracy**: Combines visual and tabular modalities for robust predictions
- **Inference Speed**: <1 second per prediction (optimized for clinical deployment)
- **Cross-Modal Learning**: Learns complementary features from both image and tabular streams

---

## 📁 Directory Structure

```
Bari/
│
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
│
├── data/
│   ├── Images/
│   │   ├── Anemic/                   # Eyelid images of anemic patients
│   │   └── Non-anemic/               # Eyelid images of healthy individuals
│   └── Tabular/
│       └── anemia.csv                # Clinical data (Age, Gender, Hb levels)
│
├── Notebook/
│   ├── Bari.ipynb                    # Main training & analysis notebook
│   ├── mobilenetv2_finetuned_visual_model.h5     # Fine-tuned visual model
│   ├── tabular_model_results.csv     # Tabular model evaluation results
│   ├── models/
│   │   ├── RandomForest_model.pkl    # Primary tabular model
│   │   ├── LogisticRegression_model.pkl
│   │   ├── XGBoost_model.pkl
│   │   └── tabular_model_results.pkl
│   └── results/                       # Training results & visualizations
│
└── Bari_api/
    ├── app.py                        # FastAPI server & prediction endpoint
    ├── utils.py                      # Utilities: preprocessing, Grad-CAM, recommendations
    └── requirements.txt              # API-specific dependencies
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda
- 2GB+ RAM (for model inference)

### Step 1: Clone & Navigate
```bash
cd c:\Users\USER\Capstone\Bari
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install API Dependencies
```bash
pip install -r Bari_api/requirements.txt
```

**Key Dependencies:**
- TensorFlow 2.14+ (deep learning)
- scikit-learn 1.3+ (Random Forest)
- FastAPI 0.108+ (REST API)
- opencv-python (image processing)
- numpy, pandas (data handling)

---

## 🏃 How to Run

### Option 1: Run the Prediction API

1. **Navigate to API directory:**
   ```bash
   cd Bari_api
   ```

2. **Ensure models are in the directory:**
   - `fusion_model.h5` (fusion neural network)
   - `RF_model.pkl` (Random Forest tabular model)
   - Should be in `Bari_api/` or notebook folder

3. **Start the FastAPI server:**
   ```bash
   python -m uvicorn app:app --reload --port 8000
   ```

4. **Access the API:**
   - Open browser: `http://localhost:8000/docs` (Swagger UI)
   - Or use curl/Postman to test endpoints

### Option 2: Run the Jupyter Notebook

1. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   ```

2. **Launch Notebook:**
   ```bash
   jupyter notebook Notebook/Bari.ipynb
   ```

3. **Run cells sequentially** to:
   - Load and explore data
   - Train/fine-tune models
   - Evaluate predictions
   - Generate visualizations

---

## 📡 API Endpoints

### POST `/predict/`
**Predict anemia severity from image and clinical data**

**Request Parameters:**
```json
{
  "image": "<image_file>",      // Eyelid image (multipart/form-data)
  "hb": 10.5,                    // Hemoglobin level (float)
  "age": 35,                     // Patient age (int)
  "gender": "female"             // Patient gender (string: "male" or "female")
}
```

**Response:**
```json
{
  "prediction": "Mild",
  "probabilities": [0.05, 0.65, 0.25, 0.05],
  "gradcam_max_value": 0.87,
  "nutritional_advice": "Include iron-rich foods like spinach, lentils, lean meat. Avoid tea/coffee with meals."
}
```

---

## 📈 Prediction Output Components

| Component | Description |
|-----------|-------------|
| **prediction** | Predicted anemia class (Normal, Mild, Moderate, Severe) |
| **probabilities** | Model confidence scores for each class [Normal, Mild, Moderate, Severe] |
| **gradcam_max_value** | Maximum intensity in the Grad-CAM heatmap (0-1) |
| **nutritional_advice** | Personalized dietary recommendations based on severity |

---

## 💡 Example Usage

### Using Curl
```bash
curl -X POST "http://localhost:8000/predict/" \
  -F "image=@path/to/eyelid_image.jpg" \
  -F "hb=11.0" \
  -F "age=40" \
  -F "gender=female"
```

### Using Python Requests
```python
import requests

files = {'image': open('eyelid_image.jpg', 'rb')}
data = {'hb': 11.0, 'age': 40, 'gender': 'female'}

response = requests.post('http://localhost:8000/predict/', files=files, data=data)
print(response.json())
```

---

## 🔬 Model Training

To retrain models on your data:

1. **Prepare your data:**
   - Place eyelid images in `data/Images/Anemic/` and `data/Images/Non-anemic/`
   - Place tabular data in `data/Tabular/anemia.csv` (columns: Age, Gender, Hb, Class)

2. **Open the notebook:**
   ```bash
   jupyter notebook Notebook/Bari.ipynb
   ```

3. **Execute training cells** to:
   - Load and preprocess images
   - Fine-tune MobileNetV2
   - Train Random Forest on tabular data
   - Train fusion model
   - Save models

---

## ⚠️ Clinical Disclaimer

**This system is for educational and research purposes only.**

- **NOT a medical diagnosis tool** – Use only as a preliminary screening aid
- **Requires professional validation** – Always consult qualified healthcare professionals
- **No liability** – Developers assume no responsibility for medical decisions based on this system
- **Supplementary only** – Must be used alongside, never instead of, proper medical examination

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 👥 Contributors

- **Project**: Bari Anemia Detection System
- **Institution**: Capstone Project
- **Date**: 2026

---

## 📧 Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.
│
├─ results/
│   ├─ confusion_matrix.png
│   ├─ accuracy_table.csv
│   └─ training_curves.png
│
├─ README.md
└─ LICENSE


⸻

Dataset Requirements
	•	Image Data: Inner eyelid images, organized by class (Anemic / Non-Anemic). Recommended resolution: 224x224.
	•	Tabular Data: Age, Gender, Hemoglobin levels.
	•	Dataset Link: [Insert link here]

⸻

Model Components

Component	Description
MobileNetV2	Pretrained CNN for extracting features from eyelid images
Random Forest	Generates tabular feature embeddings (age, gender, Hb)
Fusion Network	Fully-connected network combining image + tabular embeddings
Output Layer	Softmax classification for anemia severity


⸻

Training & Validation
	•	Visual Model: Fine-tuned MobileNetV2 on eyelid images
	•	Fusion Model: Tabular + image embeddings
	•	Loss Function: Sparse categorical cross-entropy
	•	Metrics: Accuracy, Confusion Matrix, Classification Report
	•	Early Stopping: Monitor validation loss

Example Table for Results:

Model Component	Dataset	Epochs	Train Accuracy	Val Accuracy	Notes
MobileNetV2 (Visual)	Images	30	0.75	0.73	Fine-tuned, first 100 layers frozen
Random Forest (Tabular)	Tabular	-	-	0.68	Predicts class probabilities
Fusion Network	Images + Tabular	25	0.78	0.75	Combines embeddings


⸻

Inference

Input:
	•	Eyelid image
	•	Age, Gender, Hemoglobin

Output:
	•	Predicted anemia class
	•	Grad-CAM heatmap (visual attention)
	•	Symptoms associated with predicted class
	•	Nutritional plan (WHO guidelines)
	•	Recommended physical activity

Example Usage:

from main_api import predict_anemia

image_path = "data/images/sample.jpg"
tabular_input = {"age": 25, "gender": "Female", "hb": 9.5}

result = predict_anemia(image_path, tabular_input)

print(result)
# Output: 
# {'class': 'Moderate Anemia', 'symptoms': [...], 'nutrition': [...], 'exercise': [...], 'heatmap': <image>}


⸻

API
	•	Runs via FastAPI
	•	Accepts POST requests with image + tabular data
	•	Returns JSON with predictions, recommendations, and Grad-CAM visualization

requirements.txt sample:

tensorflow==2.13.0
scikit-learn==1.3.0
fastapi==0.100.0
uvicorn==0.23.0
opencv-python==4.8.1
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.12.3


⸻

Demo Link

[Insert your live demo link here]

⸻

Next Steps / Recommendations
	1.	Expand dataset with more images for better generalization.
	2.	Add real-time camera inference in the API.
	3.	Integrate automated nutritional plan suggestions per WHO standards.
	4.	Deploy a full accessible mobile platform

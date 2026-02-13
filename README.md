# Bari-Hybrid Multimodal Anemia Detection System

A state-of-the-art **Late Fusion Ensemble** system that combines tabular blood test data (CBC) with eye conjunctiva images for accurate anemia diagnosis.

##  Architecture

### Late Fusion (Ensemble) Approach

```
┌─────────────────────┐         ┌─────────────────────┐
│  Tabular Data (CBC) │         │  Eye Image          │
│  14 Blood Features  │         │  (Conjunctiva)      │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           v                               v
┌─────────────────────┐         ┌─────────────────────┐
│   XGBoost Model     │         │  EfficientNet-B0    │
│   (GBDT)            │         │  (Pre-trained CNN)  │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │  Prob Vector (9 classes)      │
           └───────────┬───────────────────┘
                       │
                       v
              ┌────────────────┐
              │  Soft Voting   │
              │  (Weighted Avg)│
              └────────┬───────┘
                       │
                       v
              ┌────────────────┐
              │ Final Diagnosis│
              │   (9 classes)  │
              └────────────────┘
```

### Model Components

1. **Tabular Model: XGBoost**
   - Gradient Boosted Decision Trees
   - Processes 14 CBC blood test features
   - Optimized for multiclass classification
   - Feature scaling with StandardScaler

2. **Visual Model: EfficientNet-B0**
   - Pre-trained on ImageNet
   - Transfer learning with custom classification head
   - Input: 224x224 RGB images of eye conjunctiva
   - Data augmentation for robustness

3. **Late Fusion**
   - Soft Voting: Average of probability vectors
   - Configurable fusion weights (default: 0.5, 0.5)
   - Combines strengths of both modalities

##  Dataset Requirements

### Tabular Data (`diagnosed_cbc_data_v4 (1).csv`)

**Required Features (14):**
- `WBC` - White Blood Cell count
- `LYMp` - Lymphocyte percentage
- `NEUTp` - Neutrophil percentage
- `LYMn` - Lymphocyte number
- `NEUTn` - Neutrophil number
- `RBC` - Red Blood Cell count
- `HGB` - Hemoglobin
- `HCT` - Hematocrit
- `MCV` - Mean Corpuscular Volume
- `MCH` - Mean Corpuscular Hemoglobin
- `MCHC` - Mean Corpuscular Hemoglobin Concentration
- `PLT` - Platelet count
- `PDW` - Platelet Distribution Width
- `PCT` - Plateletcrit

**Target Variable:** `Diagnosis`

**9 Diagnosis Classes:**
0. Healthy
1. Iron deficiency anemia
2. Normocytic hypochromic anemia
3. Normocytic normochromic anemia
4. Other microcytic anemia
5. Thrombocytopenia
6. Leukemia
7. Leukemia with thrombocytopenia
8. Macrocytic anemia

### Image Data (`Anemia_palpebral/`)

**Directory Structure:**
```
Anemia_palpebral/
├── Healthy/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Iron deficiency anemia/
│   ├── image001.jpg
│   └── ...
├── Normocytic hypochromic anemia/
├── Normocytic normochromic anemia/
├── Other microcytic anemia/
├── Thrombocytopenia/
├── Leukemia/
├── Leukemia with thrombocytopenia/
└── Macrocytic anemia/
```

**Image Requirements:**
- Format: JPG, PNG
- Content: Eye palpebral conjunctiva
- Will be resized to 224x224 during preprocessing

##  Quick Start

### Installation

```bash
# Clone repository
git clone <repo_url>
cd multimodal-anemia-detection

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python multimodal_anemia_detection.py
```

This will:
1. Load and preprocess the CBC dataset
2. Train the XGBoost tabular model
3. Train the EfficientNet visual model
4. Save both models to `./models/`
5. Create a multimodal ensemble detector

**Expected Training Time:**
- Tabular model: ~2-5 minutes
- Visual model: ~30-60 minutes (on GPU)

### Inference

#### Single Patient Prediction

```python
from inference import AnemiaDetectorInference

# Initialize inference system
inferencer = AnemiaDetectorInference(model_dir='./models')

# Prepare patient data
tabular_data = {
    'WBC': 6.5, 'LYMp': 30.2, 'NEUTp': 58.3,
    'LYMn': 2.0, 'NEUTn': 3.8, 'RBC': 4.2,
    'HGB': 11.5, 'HCT': 35.2, 'MCV': 85.0,
    'MCH': 27.4, 'MCHC': 32.7, 'PLT': 180.0,
    'PDW': 12.5, 'PCT': 0.18
}

image_path = 'patient_eye_image.jpg'

# Get prediction
result = inferencer.predict_single(tabular_data, image_path)

print(f"Diagnosis: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Batch Prediction (CLI)

```bash
# Get model info
python inference.py --mode info --model-dir ./models

# Single prediction
python inference.py --mode single \
    --model-dir ./models \
    --image patient_eye.jpg \
    --tabular patient_data.json

# Batch prediction
python inference.py --mode batch \
    --model-dir ./models \
    --csv patients.csv \
    --image-dir patient_images/ \
    --output predictions.csv
```

##  Project Structure

```
BARI/
├── Notebook/ 
    ├── Bari.ipynb                     
├── requirements.txt                 
├── README.md                       
├── models/                         
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── efficientnet_best.h5
│   ├── efficientnet_model.h5
│   └── training_history.png
├── data/
├── Anemia_tabular/
    ├── diagnosed_cbc_data_v4 (1).csv  
├── Images/              
    ├── test
    ├── train
    
    
```

##  Advanced Usage

### Custom Fusion Weights

```python
from multimodal_anemia_detection import MultimodalAnemiaDetector

# Create detector with custom weights
# (70% tabular, 30% visual)
detector = MultimodalAnemiaDetector(
    tabular_model=tabular_model,
    visual_model=visual_model,
    fusion_weights=(0.7, 0.3)
)
```

### Model Fine-Tuning

```python
# After initial training, fine-tune EfficientNet
visual_model.fine_tune(
    train_generator=train_gen,
    val_generator=val_gen,
    epochs=10
)
```

### Evaluation

```python
# Evaluate multimodal performance
results = detector.evaluate_multimodal(
    tabular_test=X_test,
    image_test=images_test,
    y_test=y_test
)

print(f"Multimodal Accuracy: {results['accuracy']:.4f}")

# Plot confusion matrix
detector.plot_confusion_matrix(results['confusion_matrix'])
```

### Visualize Predictions

```python
# Visualize a single prediction
detector.visualize_prediction(
    tabular_input=X_test[0],
    image_input=images_test[0],
    true_label='Iron deficiency anemia',
    save_path='prediction_viz.png'
)
```

##  Performance Metrics

The system outputs:
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, Recall, F1-score
- **Confusion Matrix**: Visualization of predictions
- **Top-3 Accuracy**: Model's confidence in top 3 classes
- **Individual model contributions**: Separate tabular and visual predictions

##  Model Configuration

### XGBoost Hyperparameters

```python
params = {
    'objective': 'multi:softprob',
    'num_class': 9,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

### EfficientNet Configuration

```python
- Base Model: EfficientNet-B0 (pre-trained on ImageNet)
- Input Size: 224x224x3
- Classification Head:
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(256, ReLU, L2=0.01)
  - BatchNormalization
  - Dropout(0.5)
  - Dense(9, Softmax)
- Optimizer: Adam (lr=1e-3 → 1e-5 for fine-tuning)
- Loss: Sparse Categorical Crossentropy
```

##  Data Preprocessing

### Tabular Data
1. **Feature Extraction**: Select 14 CBC features
2. **Missing Value Handling**: Impute with column means
3. **Label Encoding**: Map diagnosis strings to integers (0-8)
4. **Feature Scaling**: StandardScaler (zero mean, unit variance)
5. **Train/Val/Test Split**: 65/15/20 with stratification

### Image Data
1. **Resizing**: All images → 224x224
2. **Normalization**: Pixel values → [0, 1]
3. **Data Augmentation** (Training only):
   - Random rotation (±20°)
   - Width/height shift (±20%)
   - Horizontal flip
   - Zoom (±15%)
   - Brightness adjustment (±20%)

##  Use Cases

1. **Clinical Decision Support**: Assist physicians in anemia diagnosis
2. **Screening Programs**: Rapid, non-invasive anemia detection
3. **Telemedicine**: Remote diagnosis using mobile eye images
4. **Research**: Study correlations between CBC data and visual symptoms
5. **Medical Education**: Teaching tool for anemia diagnosis

## [Video Demo](https://www.loom.com/share/ca95f81569b747c8a631469757e82b0b)

## [Design] (https://www.figma.com/make/x5FNIZsA2fSg32gtSBGLJ3/Bari-Healthcare-App-Prototype?t=ap3E9oC06VrRMFFs-1)

##  Important Notes

### Production Considerations
- This is a **prototype** system for initial product development
- Requires validation on diverse patient populations
- Should be used as a **decision support tool**, not standalone diagnosis
- Regulatory approval needed for clinical deployment
- Consider patient privacy and data security (HIPAA compliance)

### Model Limitations
- Performance depends on image quality (lighting, focus)
- May not generalize to severe cases not in training data
- Requires both CBC data and eye images (not robust to missing modalities)
- Class imbalance may affect rare diagnosis performance

### Future Improvements
- [ ] Add attention mechanisms for explainability
- [ ] Implement uncertainty quantification
- [ ] Support missing modality scenarios
- [ ] Add more diverse training data
- [ ] Optimize inference speed for mobile deployment
- [ ] Implement model monitoring and drift detection

##  References

### EfficientNet
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

### XGBoost
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

### Multimodal Fusion
- Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2018). Multimodal Machine Learning: A Survey and Taxonomy. TPAMI.
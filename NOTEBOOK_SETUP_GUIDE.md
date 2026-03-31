# Bari Notebook Setup Guide

## Current Status ✅ / ⚠️

### Environment
- **Location**: `c:\Users\USER\Capstone\Bari`
- **Environment Type**: Python venv
- **Python Version**: 3.14.3
- **Notebook**: `Notebook/Bari1.ipynb` (82 cells, 3,198 lines)

### Installed Packages ✓

#### Data Science & ML
- ✅ **numpy** - Numerical computing
- ✅ **pandas** - Data manipulation
- ✅ **scikit-learn** - Machine learning (Random Forest, XGBoost, Logistic Regression)
- ✅ **keras** - Neural networks (can load pre-trained .h5 models)

#### Image & Data Processing  
- ✅ **pillow** - Image manipulation
- ✅ **opencv-python-headless** - Computer vision (image resizing, preprocessing)
- ✅ **joblib** - Parallel processing & model serialization

#### API & Web Server
- ✅ **fastapi** - REST API framework
- ✅ **uvicorn** - ASGI server
- ✅ **python-multipart** - Multipart form data
- ✅ **pydantic** - Data validation

### Known Issue ⚠️

**TensorFlow is NOT installed** due to Python 3.14.3 incompatibility:
- TensorFlow <=2.15 requires Python ≤3.12
- Python 3.14.3 is too new
- Error message: "No compatible version available"

## What You Can Do NOW ✓

1. **Run All Non-TensorFlow Cells**
   - Data loading & exploration (cells 5-26)
   - Tabular model training: Random Forest, XGBoost, Logistic Regression (cells 28-43)
   - Model evaluation & comparison (cells 44-82)
   - Visualization & reporting

2. **Load Pre-Trained Models**
   - Use Keras to load `.h5` models: `keras.models.load_model('path.h5')`
   - Models available in: `Notebook/models/`
     - `fusion_model_tabular_visual.h5`
     - `mobilenetv2_finetuned_visual_model.h5`

3. **Inference on Pre-Trained Models**
   - Load visual/fusion models with Keras
   - Preprocess images with OpenCV
   - Run predictions without retraining

## Solutions for TensorFlow ⚙️

### Option 1: Switch to Python 3.11 or 3.12 (Recommended)
Create a new venv with compatible Python version:
```bash
# Windows PowerShell
python3.11 -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

### Option 2: Skip TensorFlow Now, Add Later
- Continue analysis with installed packages
- Install TensorFlow when Python 3.11/3.12 is available
- Pre-trained .h5 models work with Keras alone

### Option 3: Use TensorFlow Lite Models
- Pre-converted `.tflite` models in `models/saved_models/`
- Use `tensorflow-lite` (smaller) instead of full TensorFlow
- Python 3.14 may have some support via lite runtime

## How to Get Started

1. **Run the notebook** (start from cell 2):
   ```
   Open: Notebook/Bari1.ipynb
   Click: Run All Cells (or run cell-by-cell)
   ```

2. **Expected Results**:
   - Cell 4 may error (imports TensorFlow) - skip it
   - Cells 5+ should work fine
   - You'll see:
     - Data statistics (Cell 5)
     - Feature distributions (Cells 7-10)
     - Model comparison plots (Cells 28+)
     - Classification reports with precision/recall metrics

3. **To Use Pre-Trained Models**:
   ```python
   from keras.models import load_model
   
   # Load a pre-trained model
   model = load_model('Notebook/models/mobilenetv2_finetuned_visual_model.h5')
   predictions = model.predict(preprocessed_image)
   ```

## Project Structure

```
├── Notebook/
│   ├── Bari1.ipynb                 # Main analysis (82 cells)
│   ├── models/                     # Pre-trained models
│   │   ├── fusion_model_tabular_visual.h5
│   │   ├── mobilenetv2_finetuned_visual_model.h5
│   │   └── fusion_model_tabular_visual_with_plan.h5
│   └── results/                    # Analysis outputs
├── models/saved_models/            # TFLite models
│   ├── visual_model.tflite
│   ├── multimodal_model.tflite
│   └── multimodal_no_hb_model.tflite
├── app/                            # FastAPI application
│   ├── main.py
│   ├── services/inference.py
│   ├── utils/
│   └── schemas/
└── requirements.txt                # All dependencies defined
```

## Next Steps

1. **Immediate**: Run Notebook/Bari1.ipynb with available packages
2. **Short-term**: Switch to Python 3.11/3.12 to enable TensorFlow
3. **Long-term**: Deploy FastAPI app with model inference

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
- This is expected on Python 3.14.3
- See "Solutions for TensorFlow" above

### "ModuleNotFoundError" for other packages
- Already installed: numpy, pandas, sklearn, keras, opencv, pillow, fastapi, pydantic
- Run: `pip list` in terminal to verify

### Notebook kernel keeps restarting
- This happens during package installation
- It's normal - kernel restarts after each install batch
- Just run cells again after restart

---

**For questions about**: data preprocessing, model training, API development - everything works except TensorFlow!

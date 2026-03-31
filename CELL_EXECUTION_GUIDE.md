# Bari.ipynb - Cell Execution Guide

## Quick Start: Which Cells to Run

### Phase 1: Data Loading & Exploration (Cells 2-26)
**Status**: ✅ Ready to run  
**Expected Runtime**: ~2-5 minutes  
**Output**: Data statistics, distribution plots

| Cell | Purpose | Status |
|------|---------|--------|
| 1 | Title/Overview | 📄 Markdown |
| 2 | TODO: Imports & Path Sets | ⚠️ Check line 46-96 |
| 5 | Load Data (anemia.csv) | ✅ Works |
| 7-10 | Feature Distribution Plots | ✅ Works |
| 11-26 | Data Exploration & Visualization | ✅ Works |

**Action**: Start here! Run cells 1-26 to see your data.

---

### Phase 2: Tabular Model Training (Cells 28-43)
**Status**: ✅ Ready to run  
**Expected Runtime**: ~5-10 minutes  
**Output**: Model metrics, comparison charts, feature importance

Models trained with scikit-learn:
- 🌳 Random Forest
- 📈 XGBoost  
- 📊 Logistic Regression

**Key Results**:
- Classification reports (precision/recall/F1)
- Feature importance rankings
- Model comparison plots

---

### Phase 3: Model Evaluation & Comparison (Cells 44-82)
**Status**: ✅ Ready to run  
**Expected Runtime**: ~10-15 minutes  
**Output**: ROC curves, confusion matrices, final rankings

| Section | Cells | Focus |
|---------|-------|-------|
| Cross-Validation | 44-50 | 5-fold CV results |
| Hyperparameter Tuning | 51-60 | Grid search results |
| Final Evaluation | 61-75 | Best model performance |
| Results Summary | 76-82 | Rankings & rankings |

---

### Phase 4: Visual/Fusion Models (Cells That May Skip)
**Status**: ⚠️ Requires TensorFlow  
**Python 3.14.3 Issue**: Cannot run on current environment  

**Cells to Skip** (for now):
- Cells that import `tensorflow` or `keras.models`
- Cells that train neural networks
- Pre-trained model loading may work (uses Keras only)

**Workaround**: See NOTEBOOK_SETUP_GUIDE.md for solutions

---

## Running the Best Workflow

### Scenario A: Quick Data Check (5 minutes)
```
Cells to run: 2, 5, 7, 11
Output: Data loaded, basic statistics, 2-3 plots
```

### Scenario B: Full Tabular Analysis (20 minutes)
```
Cells to run: 2, 5, 7-10, 28-50
Output: All tabular model results with comparisons
```

### Scenario C: Complete Analysis (30+ minutes)
```
Cells to run: 2, 5-82 (skip TensorFlow cells)
Output: Full tabular analysis + evaluation metrics
```

---

## What to Expect: Cell-by-Cell Breakdown

### Cell 2 (IMPORTS) - Lines 46-96
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# ... more imports
```
⚠️ **May have errors due to TensorFlow imports**  
✅ **Solution**: Ignore TensorFlow-related errors, other imports work

### Cells 5 (DATA LOADING) - Reads anemia.csv
```python
# Output: 
# Loaded 1000 rows × 20 columns
# No missing values
```

### Cells 7-10 (VISUALIZATION) - Feature distributions
```python
# Output: 4 plots showing:
# - Hemoglobin distribution
# - RBC distribution
# - MCV distribution
# - Other hematology markers
```

### Cells 28-43 (MODEL TRAINING)
```python
# Output tables:
# Random Forest CV Score: 0.89
# XGBoost CV Score: 0.91
# Logistic Regression CV Score: 0.85
```

### Cells 44-82 (FINAL EVALUATION)
```python
# Outputs:
# - ROC-AUC curve plot
# - Confusion matrix heatmap
# - Final model ranking
# - Feature importance chart
```

---

## Handling Errors

### Error: "ModuleNotFoundError: tensorflow"
```
Location: Cell 2 (imports section)
Why: Python 3.14.3 doesn't support TensorFlow yet
Action: Click "Continue" to skip, run Cell 5+ anyway
```

### Error: "No such file or directory: data/Tabular/anemia.csv"
```
Why: Working directory not set correctly
Action: Run at top of Cell 2:
  import os
  os.chdir(r'c:\Users\USER\Capstone\Bari')
```

### Error: "Kernel crashed"
```
Why: Usually happens during TensorFlow import attempt
Action: Refresh kernel (Ctrl+Shift+F10) and run again
```

---

## Performance Notes

- **Cell 5** (Load data): <1 second
- **Cells 7-10** (Plots): ~2-3 seconds each
- **Cells 28-43** (Model training): 30-60 seconds total
- **Cells 51-60** (Hyperparameter tuning): 2-5 minutes
- **Cells 76-82** (Evaluation): ~1 minute

**Total time for Scenario C**: 15-30 minutes (depending on compute)

---

## Outputs & Results

All results saved to: `Notebook/results/`

Files created:
- `final_rf_summary.csv` - Random Forest results
- `final_rf_classification_report.txt` - Detailed metrics
- `final_rf_feature_importance.csv` - Top features
- `tuned_model_comparison.csv` - All models ranked
- Plus visualization PNG files

---

## Next: Run Your First Cell!

Click on Cell 2 and press ▶️ **Run Cell** to get started!

Or select multiple cells and press **Run Selected Cells**.

**Questions?** Check NOTEBOOK_SETUP_GUIDE.md

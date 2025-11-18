# 🚀 Quick Start Guide

Get started with Software Defect Prediction in 5 minutes!

## ⚡ Super Quick Start

```bash
# 1. Clone repository
git clone https://github.com/kukuhyudhistiro/ase_35metrics.git
cd ase_35metrics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python src/sdp_jurnal_improved.py

# 4. View results
# - Check console output for metrics
# - Open outputs/sdp_comprehensive_analysis.png
# - Review CSV files in outputs/
```

**Expected Runtime**: 2-3 minutes  
**Expected Output**: Perfect classification (100% all metrics) ✅

---

## 📋 Step-by-Step Guide

### Step 1: Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
# Test imports
python -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
print('✅ All dependencies installed successfully!')
"
```

### Step 3: Run Analysis

```bash
# Navigate to source directory
cd src

# Run complete analysis
python sdp_jurnal_improved.py
```

**Console Output Sample:**
```
================================================================================
SOFTWARE DEFECT PREDICTION - COMPREHENSIVE ANALYSIS
================================================================================

[1] Dataset loaded successfully: (500, 35)

[2.1] Dataset Overview:
  - Number of samples: 500
  - Number of features: 34
  - Missing values: 0

[2.2] Class Distribution Analysis:
  - Non-defective (0): 376 (75.20%)
  - Defective (1): 124 (24.80%)
  - Imbalance Ratio: 3.03:1

...

[4.2] Model Training & Evaluation:

  → Training Random Forest + SMOTE...
     Accuracy:  1.0000
     Precision: 1.0000
     Recall:    1.0000
     F1-Score:  1.0000
     AUC-ROC:   1.0000

✓ Comprehensive visualization saved to: outputs/sdp_comprehensive_analysis.png
```

### Step 4: Explore Results

```bash
# View visualization
open outputs/sdp_comprehensive_analysis.png  # Mac
xdg-open outputs/sdp_comprehensive_analysis.png  # Linux
start outputs/sdp_comprehensive_analysis.png  # Windows

# Check CSV results
cat outputs/model_comparison.csv
cat outputs/feature_importance.csv
cat outputs/cv_scores.csv
```

---

## 🔍 Understanding the Results

### Confusion Matrix
```
              Predicted
              0    1
Actual  0  [113   0]  ← Perfect! No false positives
        1  [  0  37]  ← Perfect! No false negatives
```

### Feature Importance
```
Top 5 Features:
1. scc (0.5596) - 56% of prediction power! ⭐
2. dcl (0.0492) - 5%
3. prm (0.0257) - 3%
4. ccr (0.0252) - 3%
5. l   (0.0213) - 2%
```

### What This Means

✅ **Perfect Classification**: Model catches ALL defects (37/37)  
✅ **No False Alarms**: Zero unnecessary investigations  
✅ **Sum Cyclomatic Complexity** is the key predictor  
✅ **Actionable**: Modules with scc > 45 need refactoring  

---

## 🎯 Common Use Cases

### Use Case 1: Quick Defect Check

```python
import pandas as pd
import joblib

# Load trained model (after first run)
model = joblib.load('outputs/rf_model.pkl')
scaler = joblib.load('outputs/scaler.pkl')

# Prepare your module metrics
my_module = {
    'scc': 52,  # High! Refactor needed
    'dcl': 120,
    'prm': 4,
    'ccr': 0.35,
    # ... other 30 metrics
}

# Predict
X = pd.DataFrame([my_module])
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)

if prediction[0] == 1:
    print("⚠️ DEFECT PREDICTED - Prioritize for review!")
else:
    print("✅ No defect predicted")
```

### Use Case 2: Batch Analysis

```python
# Analyze multiple modules
modules_df = pd.read_csv('your_modules.csv')
X = modules_df.drop('defects', axis=1)
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

# Get high-risk modules
high_risk = modules_df[predictions == 1]
print(f"Found {len(high_risk)} high-risk modules")
print(high_risk[['module_name', 'scc', 'dcl']])
```

### Use Case 3: Feature Analysis

```python
# Check which metrics are problematic in your code
feature_importance = pd.read_csv('outputs/feature_importance.csv')
my_metrics = pd.Series(my_module)

# Compare with thresholds
thresholds = {
    'scc': 45,
    'dcl': 150,
    'prm': 6,
    'ccr': (0.15, 0.7)
}

for metric, threshold in thresholds.items():
    value = my_metrics[metric]
    if metric == 'ccr':
        if value < threshold[0] or value > threshold[1]:
            print(f"⚠️ {metric} = {value} (outside optimal range)")
    else:
        if value > threshold:
            print(f"⚠️ {metric} = {value} (exceeds threshold {threshold})")
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: "File not found"

```bash
# Ensure you're in the correct directory
pwd  # Should show /path/to/ase_35metrics

# Check data file exists
ls data/datasetjurnal_lengkap_500.csv
```

### Issue: "Memory error"

```python
# Reduce visualization resolution in sdp_jurnal_improved.py
# Change line:
plt.savefig('output.png', dpi=300)  # Original
plt.savefig('output.png', dpi=150)  # Reduced
```

### Issue: "SHAP takes too long"

```python
# Reduce sample size for SHAP
shap_sample = X_test_scaled[:100]  # Only first 100 samples
shap_values = explainer.shap_values(shap_sample)
```

---

## 📚 Next Steps

After completing the quick start:

1. **Read the Full Documentation**
   - [README.md](README.md) - Complete project overview
   - [Methodology](docs/methodology.md) - Detailed methods
   - [Results](docs/results.md) - Full result analysis

2. **Explore Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

3. **Customize for Your Project**
   - Replace dataset with your own
   - Adjust hyperparameters
   - Add new metrics

4. **Integrate into CI/CD**
   - See [Integration Guide](docs/integration.md)
   - Use provided quality gates
   - Set up automated checks

5. **Contribute**
   - Report issues
   - Suggest improvements
   - Share your results

---

## 💡 Pro Tips

### Tip 1: Save Trained Model
```python
import joblib

# After training
joblib.dump(rf_model, 'outputs/rf_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')

# For reuse
model = joblib.load('outputs/rf_model.pkl')
```

### Tip 2: Custom Thresholds
```python
# Adjust prediction threshold for precision/recall trade-off
y_proba = model.predict_proba(X_test_scaled)[:, 1]
custom_threshold = 0.3  # Lower = more sensitive

predictions = (y_proba > custom_threshold).astype(int)
```

### Tip 3: Export for Excel
```python
# Create detailed report
import pandas as pd

results = pd.DataFrame({
    'Module': module_names,
    'Prediction': predictions,
    'Probability': y_proba,
    'scc': X_test['scc'],
    'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                   for p in y_proba]
})

results.to_excel('outputs/defect_report.xlsx', index=False)
```

---

## 🎓 Learning Resources

- **SMOTE**: [Original Paper](https://arxiv.org/abs/1106.1813)
- **Random Forest**: [Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324)
- **SHAP**: [Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874)
- **Software Metrics**: [IEEE Standards](https://standards.ieee.org/)

---

## 📞 Get Help

- 💬 [GitHub Discussions](https://github.com/kukuhyudhistiro/ase_35metrics/discussions)
- 🐛 [Report Issues](https://github.com/kukuhyudhistiro/ase_35metrics/issues)
- 📧 Email: kukuh.yudhistiro@example.com

---

**Ready to predict defects like a pro?** 🚀

[← Back to README](README.md) | [View Full Documentation →](docs/)

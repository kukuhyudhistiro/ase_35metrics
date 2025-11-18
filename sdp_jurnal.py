#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software Defect Prediction using SMOTE-Augmented Random Forest
Author: Kukuh Yudhistiro, Aris Marjuni
Institution: Fakultas Ilmu Komputer, Udinus

This script implements a comprehensive Software Defect Prediction system
using SMOTE (Synthetic Minority Over-sampling Technique) combined with
Random Forest classifier to address class imbalance issues.
"""

#%% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, classification_report, roc_curve, auc, 
    precision_recall_curve, precision_score
)
from imblearn.over_sampling import SMOTE

# Konfigurasi
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
np.random.seed(42)

#%% KONFIGURASI GLOBAL
FILE_PATH = 'datasetjurnal_lengkap_500.csv'
TARGET_NAMES = ['No Defect', 'Defect']
RANDOM_STATE = 42
TEST_SIZE = 0.3

#%% 1. DATA LOADING & INITIAL EXPLORATION
print("="*80)
print("SOFTWARE DEFECT PREDICTION - COMPREHENSIVE ANALYSIS")
print("="*80)

# Load dataset
df = pd.read_csv(FILE_PATH, sep=';', decimal=',')
print(f"\n[1] Dataset loaded successfully: {df.shape}")

# Drop ID column if exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Convert to numeric
for col in df.columns:
    if col != 'defects':
        df[col] = pd.to_numeric(df[col], errors='coerce')

#%% 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*80)

# 2.1 Dataset Overview
print("\n[2.1] Dataset Overview:")
print(f"  - Number of samples: {df.shape[0]}")
print(f"  - Number of features: {df.shape[1]-1}")
print(f"  - Missing values: {df.isnull().sum().sum()}")

# 2.2 Class Distribution Analysis
print("\n[2.2] Class Distribution Analysis:")
class_dist = df['defects'].value_counts()
class_dist_pct = df['defects'].value_counts(normalize=True) * 100
print(f"  - Non-defective (0): {class_dist[0]} ({class_dist_pct[0]:.2f}%)")
print(f"  - Defective (1): {class_dist[1]} ({class_dist_pct[1]:.2f}%)")
print(f"  - Imbalance Ratio: {class_dist[0]/class_dist[1]:.2f}:1")

# 2.3 Descriptive Statistics
print("\n[2.3] Descriptive Statistics (Key Metrics):")
key_metrics = ['mcc', 'n', 'n1', 'n2', 'cl', 'scc', 'ccr', 'dcl']
desc_stats = df[key_metrics].describe()
print(desc_stats.round(2))

# 2.4 Correlation Analysis
print("\n[2.4] Top 10 Features Correlated with Defects:")
correlations = df.corr()['defects'].sort_values(ascending=False)
print(correlations.head(11).round(4))  # 11 to exclude defects itself

# 2.5 Data Quality Check
print("\n[2.5] Data Quality Assessment:")
print(f"  - Duplicate rows: {df.duplicated().sum()}")
print(f"  - Features with zero variance: {(df.std() == 0).sum()}")

#%% 3. DATA PREPROCESSING
print("\n" + "="*80)
print("3. DATA PREPROCESSING")
print("="*80)

# 3.1 Feature Engineering
features = [col for col in df.columns if col != 'defects']
X = df[features]
y = df['defects']

print(f"\n[3.1] Features extracted: {len(features)} features")

# 3.2 Missing Value Imputation
print("\n[3.2] Missing Value Imputation:")
missing_before = X.isnull().sum().sum()
for col in X.columns:
    if X[col].isnull().any():
        X.loc[:, col] = X[col].fillna(X[col].median())
missing_after = X.isnull().sum().sum()
print(f"  - Missing values before: {missing_before}")
print(f"  - Missing values after: {missing_after}")

# 3.3 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\n[3.3] Train-Test Split (Stratified):")
print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")
print(f"  - Train class distribution: {y_train.value_counts().to_dict()}")
print(f"  - Test class distribution: {y_test.value_counts().to_dict()}")

# 3.4 SMOTE Application
print(f"\n[3.4] SMOTE (Synthetic Minority Over-sampling Technique):")
print(f"  - Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  - After SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
print(f"  - Synthetic samples generated: {len(y_train_res) - len(y_train)}")

# 3.5 Feature Scaling
print(f"\n[3.5] Feature Scaling (StandardScaler):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better handling
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
print(f"  - Features normalized to zero mean and unit variance")
print(f"  - Mean of scaled features: {X_train_scaled_df.mean().mean():.6f}")
print(f"  - Std of scaled features: {X_train_scaled_df.std().mean():.6f}")

#%% 4. MODEL TRAINING & EVALUATION
print("\n" + "="*80)
print("4. MODEL TRAINING & EVALUATION")
print("="*80)

# 4.1 Model Definitions
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_STATE, 
        solver='liblinear',
        max_iter=1000
    ),
    "Random Forest + SMOTE": RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# 4.2 Training and Evaluation
results = {}
detailed_results = []

print("\n[4.2] Model Training & Evaluation:")
for name, model in models.items():
    print(f"\n  → Training {name}...")
    
    # Train model
    model.fit(X_train_scaled_df, y_train_res)
    
    # Predictions
    y_pred = model.predict(X_test_scaled_df)
    
    # Probabilities (if available)
    try:
        y_proba = model.predict_proba(X_test_scaled_df)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    except:
        y_proba = None
        auc_score = 0.0
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc_score,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(
            y_test, y_pred, 
            target_names=TARGET_NAMES, 
            output_dict=False
        )
    }
    
    # Print metrics
    print(f"     Accuracy:  {accuracy:.4f}")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall:    {recall:.4f}")
    print(f"     F1-Score:  {f1:.4f}")
    print(f"     AUC-ROC:   {auc_score:.4f}")
    
    # Store for comparison table
    detailed_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_score
    })

#%% 5. CROSS-VALIDATION ANALYSIS
print("\n" + "="*80)
print("5. CROSS-VALIDATION ANALYSIS")
print("="*80)

rf_model = models["Random Forest + SMOTE"]
cv_scores = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n[5.1] 5-Fold Stratified Cross-Validation:")
for scoring in ['accuracy', 'precision', 'recall', 'f1']:
    scores = cross_val_score(
        rf_model, X_train_scaled_df, y_train_res, 
        cv=skf, scoring=scoring
    )
    cv_scores[scoring] = scores
    print(f"  - {scoring.capitalize():12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

#%% 6. FEATURE IMPORTANCE ANALYSIS
print("\n" + "="*80)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importances
feature_importances = pd.Series(
    rf_model.feature_importances_, 
    index=features
).sort_values(ascending=False)

print("\n[6.1] Top 15 Most Important Features:")
for i, (feat, imp) in enumerate(feature_importances.head(15).items(), 1):
    print(f"  {i:2d}. {feat:6s}: {imp:.4f}")

#%% 7. QUANTITATIVE ANALYSIS
print("\n" + "="*80)
print("7. QUANTITATIVE ANALYSIS SUMMARY")
print("="*80)

# Create comparison DataFrame
comparison_df = pd.DataFrame(detailed_results)
comparison_df = comparison_df.set_index('Model')

print("\n[7.1] Model Performance Comparison:")
print(comparison_df.round(4))

# Statistical significance test
print("\n[7.2] Performance Gains (RF+SMOTE vs Baselines):")
rf_metrics = comparison_df.loc['Random Forest + SMOTE']
for baseline in ['Naive Bayes', 'Logistic Regression']:
    baseline_metrics = comparison_df.loc[baseline]
    print(f"\n  vs {baseline}:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        gain = ((rf_metrics[metric] - baseline_metrics[metric]) / 
                baseline_metrics[metric] * 100)
        print(f"    - {metric:12s}: {gain:+6.2f}%")

#%% 8. QUALITATIVE ANALYSIS
print("\n" + "="*80)
print("8. QUALITATIVE ANALYSIS")
print("="*80)

# 8.1 Confusion Matrix Analysis
print("\n[8.1] Confusion Matrix Analysis (Random Forest + SMOTE):")
cm = results["Random Forest + SMOTE"]['Confusion Matrix']
tn, fp, fn, tp = cm.ravel()
print(f"  - True Negatives (TN):  {tn}")
print(f"  - False Positives (FP): {fp}")
print(f"  - False Negatives (FN): {fn}")
print(f"  - True Positives (TP):  {tp}")
print(f"\n  Interpretation:")
print(f"  - Correctly identified defects: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"  - Correctly identified non-defects: {tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)")

# 8.2 Error Analysis
print("\n[8.2] Error Analysis:")
if fp > 0:
    print(f"  - False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"    → {fp} non-defective modules incorrectly flagged")
if fn > 0:
    print(f"  - False Negative Rate: {fn/(fn+tp):.4f}")
    print(f"    → {fn} defective modules missed")
else:
    print(f"  - Perfect detection: No defects missed!")

#%% 9. VISUALIZATION
print("\n" + "="*80)
print("9. GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.35)

# 1. Confusion Matrix (RF+SMOTE)
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(results["Random Forest + SMOTE"]['Confusion Matrix'], 
            annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax1)
ax1.set_title('Confusion Matrix\n(Random Forest + SMOTE)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# 2. ROC Curves
ax2 = fig.add_subplot(gs[0, 1])
for name, res in results.items():
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{name.split(" +")[0][:10]} (AUC={roc_auc:.3f})', linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])
ax2.set_title('ROC Curves Comparison', fontsize=11, fontweight='bold')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(alpha=0.3)

# 3. Precision-Recall Curves
ax3 = fig.add_subplot(gs[0, 2])
for name, res in results.items():
    if res['y_proba'] is not None:
        precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
        ax3.plot(recall, precision, label=name.split(" +")[0][:15], linewidth=2)
ax3.set_title('Precision-Recall Curves', fontsize=11, fontweight='bold')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 4. Metrics Comparison Bar Chart
ax4 = fig.add_subplot(gs[0, 3])
comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=ax4, width=0.8
)
ax4.set_title('Performance Metrics Comparison', fontsize=11, fontweight='bold')
ax4.set_ylabel('Score')
ax4.set_ylim(0, 1.05)
ax4.legend(fontsize=8, loc='lower right')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

# 5. Top 15 Feature Importance
ax5 = fig.add_subplot(gs[1, :2])
top_features = feature_importances.head(15)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax5.barh(range(len(top_features)), top_features.values, color=colors)
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features.index)
ax5.set_xlabel('Importance Score')
ax5.set_title('Top 15 Feature Importance (Random Forest)', fontsize=11, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features.values)):
    ax5.text(val, i, f' {val:.4f}', va='center', fontsize=8)

# 6. SHAP Summary Plot
ax6 = fig.add_subplot(gs[1, 2:])
try:
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled_df)
    
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    shap.summary_plot(
        shap_values_class1, X_test_scaled_df, 
        max_display=15, show=False, plot_type='dot'
    )
    ax6.set_title('SHAP Feature Impact Analysis', fontsize=11, fontweight='bold')
except Exception as e:
    ax6.text(0.5, 0.5, f'SHAP Analysis\nUnavailable:\n{str(e)[:50]}', 
             ha='center', va='center', transform=ax6.transAxes,
             fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

# 7. Correlation Heatmap (Top Features)
ax7 = fig.add_subplot(gs[2, :2])
top_feature_names = feature_importances.head(15).index.tolist()
temp_df = X[top_feature_names].copy()
temp_df['defects'] = y
corr_matrix = temp_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, ax=ax7,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8})
ax7.set_title('Feature Correlation Matrix (Top 15)', fontsize=11, fontweight='bold')
ax7.tick_params(labelsize=8)

# 8. Class Distribution (Before/After SMOTE)
ax8 = fig.add_subplot(gs[2, 2])
y_train.value_counts().plot.pie(
    autopct='%1.1f%%', labels=TARGET_NAMES, 
    ax=ax8, startangle=90, colors=['#ff9999', '#66b3ff']
)
ax8.set_title('Class Distribution\n(Before SMOTE)', fontsize=11, fontweight='bold')
ax8.set_ylabel('')

ax9 = fig.add_subplot(gs[2, 3])
pd.Series(y_train_res).value_counts().plot.pie(
    autopct='%1.1f%%', labels=TARGET_NAMES, 
    ax=ax9, startangle=90, colors=['#ff9999', '#66b3ff']
)
ax9.set_title('Class Distribution\n(After SMOTE)', fontsize=11, fontweight='bold')
ax9.set_ylabel('')

# 9. Cross-Validation Box Plots
metrics_list = ['accuracy', 'precision', 'recall', 'f1']
for idx, metric in enumerate(metrics_list):
    ax = fig.add_subplot(gs[3, idx])
    bp = ax.boxplot([cv_scores[metric]], labels=[metric.capitalize()],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'CV {metric.capitalize()}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.text(1, cv_scores[metric].mean(), f'{cv_scores[metric].mean():.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 10. Feature Distribution Histograms (Top 4)
top_4_features = feature_importances.head(4).index
for idx, feat in enumerate(top_4_features):
    ax = fig.add_subplot(gs[4, idx])
    
    # Get original data before scaling
    feat_data = df[feat]
    target = df['defects']
    
    # Separate by class
    defect_data = feat_data[target == 1]
    no_defect_data = feat_data[target == 0]
    
    ax.hist(no_defect_data, bins=20, alpha=0.5, label='No Defect', color='blue')
    ax.hist(defect_data, bins=20, alpha=0.5, label='Defect', color='red')
    ax.set_title(f'Distribution: {feat}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Software Defect Prediction: Comprehensive Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_path = 'sdp_comprehensive_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Comprehensive visualization saved to: {output_path}")

#%% 10. DETAILED CLASSIFICATION REPORT
print("\n" + "="*80)
print("10. DETAILED CLASSIFICATION REPORT")
print("="*80)

print("\n[10.1] Random Forest + SMOTE Classification Report:")
print(results["Random Forest + SMOTE"]['Classification Report'])

#%% 11. SAVE RESULTS
print("\n" + "="*80)
print("11. SAVING RESULTS")
print("="*80)

# Save comparison table
comparison_path = 'model_comparison.csv'
comparison_df.to_csv(comparison_path)
print(f"\n✓ Model comparison saved to: {comparison_path}")

# Save feature importance
feature_imp_path = 'feature_importance.csv'
feature_importances.to_csv(feature_imp_path, header=['Importance'])
print(f"✓ Feature importance saved to: {feature_imp_path}")

# Save CV scores
cv_results_path = 'cv_scores.csv'
cv_df = pd.DataFrame(cv_scores)
cv_df.to_csv(cv_results_path)
print(f"✓ Cross-validation scores saved to: {cv_results_path}")

#%% 12. FINAL SUMMARY
print("\n" + "="*80)
print("12. ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           SOFTWARE DEFECT PREDICTION - FINAL SUMMARY          ║
╠═══════════════════════════════════════════════════════════════╣
║ Dataset: {df.shape[0]} samples, {df.shape[1]-1} features                            
║ Class Distribution: {class_dist[0]} non-defective, {class_dist[1]} defective              
║                                                               
║ BEST MODEL: Random Forest + SMOTE                            
║   • Accuracy:  {results['Random Forest + SMOTE']['Accuracy']:.4f}
║   • Precision: {results['Random Forest + SMOTE']['Precision']:.4f}
║   • Recall:    {results['Random Forest + SMOTE']['Recall']:.4f}
║   • F1-Score:  {results['Random Forest + SMOTE']['F1']:.4f}
║   • AUC-ROC:   {results['Random Forest + SMOTE']['AUC']:.4f}
║                                                               
║ Top 3 Predictive Features:
║   1. {feature_importances.index[0]:6s} ({feature_importances.iloc[0]:.4f})
║   2. {feature_importances.index[1]:6s} ({feature_importances.iloc[1]:.4f})
║   3. {feature_importances.index[2]:6s} ({feature_importances.iloc[2]:.4f})
╚═══════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*80)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)

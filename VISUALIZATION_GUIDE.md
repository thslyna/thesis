# 📊 Visualization Guide - V1_IMPUTED Results

## 🎯 KEY PERFORMANCE PLOTS (Use These in Chapter 4!)

### 1. ROC Curves (Model Performance)
**Best model:**
- `results/rf_tuned/roc_curve.png` - Random Forest (0.956 AUC) ⭐⭐⭐

**Other models:**
- `results/xgb_tuned/roc_curve.png` - XGBoost Tuned (0.931 AUC)
- `results/xgb_untuned/roc_curve.png` - XGBoost Untuned (0.903 AUC)
- `results/logreg/roc_curve.png` - Logistic Regression

**Usage:** Figure 4.1 - Model comparison via ROC curves

---

### 2. Confusion Matrices (Prediction Accuracy)
- `results/rf_tuned/confusion_matrix.png` - Best model performance
- `results/xgb_tuned/confusion_matrix.png`
- `results/xgb_untuned/confusion_matrix.png`
- `results/logreg/confusion_matrix.png`

**Usage:** Figure 4.2 - Confusion matrix showing TP/FP/TN/FN

---

### 3. Feature Importance (What Matters Most)
- `results/rf_untuned/feature_importance.png` - Shows top predictive features
- `results/logreg/feature_importance.png` - Linear model coefficients

**Usage:** Figure 4.3 - Feature importance rankings

---

## 🔬 ADVANCED ANALYSES (New from today!)

### 4. Contextual Analysis
- `results/lag_correlation.png` - Road traffic lag effects
- `results/rush_hour_analysis.png` - Time-of-day patterns  
- `results/weather_road_interaction.png` - Weather moderation

**Usage:** Section 4.X - Contextual Performance Analysis

---

### 5. Model Interpretability
- `results/shap/importance.png` - Global feature importance
- `results/shap/summary.png` - Feature impact distribution

**Usage:** Section 4.Y - Model Interpretability (SHAP)

---

### 6. Temporal Validation
- `results/temporal_validation/temporal_validation.png` - Oct-only vs Full model

**Usage:** Section 4.Z - Temporal Robustness

---

### 7. Economic Impact
- `results/cost_benefit/cost_benefit_analysis.png` - €6.15M annual savings

**Usage:** Section 4.W - Operational Business Value

---

### 8. Calibration & Thresholds
- `results/v1_calibration/calibration_curve_rf_v1.png` - Probability calibration
- `results/v1_threshold_analysis/precision_recall_vs_threshold.png`
- `results/v1_threshold_analysis/threshold_error_tradeoff.png`

**Usage:** Section 4.5 - Model Calibration and Threshold Analysis

---

### 9. Ablation Study
- `results/v1_ablation/ablation_false_negatives.png` - Feature group importance

**Usage:** Section 4.4 - Ablation Study

---

## 📋 THESIS FIGURE PLAN

**Chapter 4 Figures (Recommended Order):**

1. **Figure 4.1:** ROC Curves (rf_tuned/roc_curve.png) - 0.956 AUC!
2. **Figure 4.2:** Confusion Matrix (rf_tuned/confusion_matrix.png)
3. **Figure 4.3:** Feature Importance (rf_untuned/feature_importance.png)
4. **Figure 4.4:** Calibration Curve (v1_calibration/calibration_curve_rf_v1.png)
5. **Figure 4.5:** Ablation Study (v1_ablation/ablation_false_negatives.png)
6. **Figure 4.6:** Lag Correlation (lag_correlation.png)
7. **Figure 4.7:** Time-of-Day Analysis (rush_hour_analysis.png)
8. **Figure 4.8:** Weather Interaction (weather_road_interaction.png)
9. **Figure 4.9:** SHAP Summary (shap/summary.png)
10. **Figure 4.10:** Temporal Validation (temporal_validation/temporal_validation.png)
11. **Figure 4.11:** Cost-Benefit (cost_benefit/cost_benefit_analysis.png)

**Total: 11 publication-quality figures!** ✅

---

## 🎨 How to Include in Thesis (LaTeX)
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results/rf_tuned/roc_curve.png}
\caption{ROC curves comparing model performance. Random Forest (tuned) achieved 0.956 AUC on 8,291 flights.}
\label{fig:roc_curves}
\end{figure}
```

---

## 📊 Quick Stats Summary

**Dataset:** 8,291 flights (Oct-Nov-Jan-Feb 2025-2026)
**Best Model:** Random Forest (tuned)
**Performance:** 0.956 AUC (+9.4% vs 4,330 flights)
**Weather Coverage:** 93.5% (via 6-hour imputation)
**Economic Impact:** €6.15M annual savings

---

**All visualizations ready for your thesis!** 🎓

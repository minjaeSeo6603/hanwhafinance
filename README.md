# Hanwha Investment & Securities - AI Data Scientist Case Study

## Statement of the Problem

Develop a **binary classification model** to predict whether retail banking customers will default on loans, addressing:

1. **Feature Selection**: Among 100+ correlated features, identify truly predictive variables using L1 regularization
2. **Class Imbalance**: Default rate is only 5.4%; traditional accuracy metrics fail. Evaluate via F1-Score and AUCPR
3. **Production Monitoring**: Model performs well offline but degrades in production. Detect and respond to concept drift

---

## Project Structure

```
.
├── README.md                                  # This file
│
├── code/                                      # Source code (Jupyter notebooks)
│   ├── Q1_Feature_Selection.ipynb            # Lasso & multicollinearity analysis
│   ├── Q2_Class_Imbalance.ipynb              # Class imbalance & evaluation metrics
│   └── Q3_Bias_Variance_Drift.ipynb          # Concept drift & retraining strategy
│
├── output/                                    # Visualization outputs
│   ├── Q1/                                   # Question 1 visualizations
│   │   ├── l1_vs_l2_geometry.png             # L1 (diamond) vs L2 (circle) constraint geometry
│   │   ├── lasso_feature_selection.png       # Feature selection at different regularization strengths
│   │   ├── correlation_heatmap.png           # Feature correlation matrix
│   │   └── multicollinearity_comparison.png  # Logistic Regression vs Random Forest impact
│   │
│   ├── Q2/                                   # Question 2 visualizations
│   │   ├── class_imbalance.png               # Class distribution (94.6% vs 5.4%)
│   │   ├── confusion_matrix_explained.png    # TP/FP/TN/FN breakdown
│   │   ├── metrics_comparison.png            # Accuracy vs Precision vs Recall vs F1-Score
│   │   ├── roc_vs_pr_curve.png               # ROC-AUC vs Precision-Recall curve comparison
│   │   ├── smote_visualization.jpg           # SMOTE synthetic data generation in feature space
│   │   ├── model_comparison.jpg              # Baseline vs SMOTE vs Cost-Sensitive learning
│   │   └── robustness_comparison.png         # F1-Score variance across CV folds
│   │
│   └── Q3/                                   # Question 3 visualizations
│       ├── learning_curve_diagnosis.png      # High-variance vs balanced model learning curves
│       ├── bias_variance_tradeoff.png        # Polynomial degree: underfitting to overfitting
│       ├── performance_degradation.png       # Out-of-time F1-Score decay (Q1–Q2 2021)
│       ├── concept_drift_simulation.png      # Income distribution shift over 6 quarters
│       ├── psi_trend.png                     # Population Stability Index monitoring (PSI ≥ 0.25 alert)
│       └── csi_analysis.png                  # Characteristic Stability Index by feature
│
├── report/
│   ├── report.qmd                            # Quarto markdown technical report
│   └── report.pdf                            # Compiled PDF (2–3 pages)
│
├── requirements.txt                          # Python dependencies
```

---

## Case Study Questions & Solutions

### Q1: Feature Selection & Multicollinearity

**Question**: Among hundreds of features, select truly predictive variables using Lasso. How does L1 regularization work mathematically? Compare how Random Forest and Logistic Regression are affected by multicollinearity.

**Key Insights**:

- **L1 (Lasso) Mathematical Principle**: 
  - Objective: $\min_\beta \frac{1}{n}\sum_i[-y_i\log\hat{p}_i - (1-y_i)\log(1-\hat{p}_i)] + \lambda\sum_j|\beta_j|$
  - KKT condition: Features with marginal gradient $|g_j(\hat\beta)| < \lambda$ are exactly zeroed
  - Geometric insight: L1 constraint forms a **diamond** (vertices = sparsity) vs L2 **circle** (only shrinkage)

- **Multicollinearity Impact**:
  - **Logistic Regression**: Coefficients become **highly unstable**; likelihood surface flattens when features are correlated; interpretation fails
  - **Random Forest**: **Robust**; splits sequentially on features producing similar partitions; predictions remain stable
  - **Recommendation**: Use L1-penalized logistic regression for interpretability; Random Forest as validation benchmark

**Visualizations**:
- `l1_vs_l2_geometry.png`: Constraint regions and loss contours
- `lasso_feature_selection.png`: Sparsity at C = 0.01, 0.1, 1.0, 10.0
- `multicollinearity_comparison.png`: Coefficient instability vs importance stability
- `correlation_heatmap.png`: Feature correlation patterns

**Code**: `Q1_Feature_Selection.ipynb`

---

### Q2: Class Imbalance & Evaluation Metrics

**Question**: With 5.4% default rate, why is accuracy misleading? Justify F1-Score and AUCPR. Compare SMOTE vs cost-sensitive learning robustness.

**Key Insights**:

- **Accuracy Trap**: 
  - Naive "always non-default" classifier: 96.6% accuracy, 0% recall on defaulters—useless
  - **F1-Score**: Harmonic mean of precision and recall; penalizes both false negatives and false positives
  - **AUCPR**: Precision-Recall curve; baseline = class prevalence (~5.4%); directly exposes minority-class challenge

- **SMOTE vs Cost-Sensitive Learning**:
  - **SMOTE**: Generates synthetic minority samples; risks creating economically implausible borrowers and overfitting
  - **Cost-Sensitive Learning**: Weight minority errors more ($w_1 > w_0$); **preserves true distribution**; more robust under production drift
  - **Winner**: Cost-sensitive learning (F1 = 0.742, variance σ = 0.017 vs SMOTE σ = 0.020)

**Visualizations**:
- `class_imbalance.png`: 94.6% non-default vs 5.4% default
- `confusion_matrix_explained.png`: TP/FP/TN/FN breakdown
- `roc_vs_pr_curve.png`: ROC-AUC = 0.850 misleading; AUCPR = 0.612 reveals imbalance
- `metrics_comparison.png`: Accuracy (96.6%) vs F1-Score (0.564) for baseline
- `smote_visualization.png`: Synthetic data generation in PCA space
- `model_comparison.png`: Baseline vs SMOTE vs cost-sensitive F1 comparison
- `robustness_comparison.png`: F1-Score variance across 10 CV folds

**Code**: `Q2_Class_Imbalance.ipynb`

---

### Q3: Bias-Variance Trade-Off & Concept Drift

**Question**: Model works well offline but degrades in production. Diagnose via bias-variance. How to detect concept drift (PSI/CSI)? What retraining strategy?

**Key Insights**:

- **Bias-Variance Diagnosis**:
  - **High-variance (overfitting)**: Training accuracy ≈ 1.0, validation ≈ 0.90; persistent gap
  - **For credit risk**: Root cause is usually **concept drift**, not simple overfitting
  - Solution: Monitor with PSI/CSI; trigger retraining when drift detected

- **Drift Detection (PSI & CSI)**:
  - **PSI (Population Stability Index)**: $\text{PSI} = \sum_i (q_i - p_i)\log(q_i/p_i)$
    - Thresholds: PSI < 0.1 (stable), 0.1–0.25 (monitor), ≥0.25 (urgent)
  - **CSI (Characteristic Stability Index)**: Applies PSI to model outputs (score deciles); reveals feature-level drift

- **Real-World Scenario** (2020 Q1 model):
  - Q1–Q2: PSI ≈ 0 (stable)
  - Q3 2020: PSI = 0.254 (threshold breach) 
  - Q4 2020 (Recession): PSI = 1.020; F1 drops 40%
  - Q1–Q2 2021 (Crisis): PSI = 1.975; income collapsed 25%; default rate tripled

- **Retraining Strategy** (Hybrid):
  - **Quarterly scheduled**: Retrain on rolling 12–24 month window
  - **Trigger-based**: PSI ≥ 0.25 or F1 drop > 10% → emergency retraining
  - **Pre-deployment**: Backtest on holdout; confirm improvements in F1, AUCPR, PSI/CSI

**Visualizations**:
- `learning_curve_diagnosis.png`: Overfitted (persistent gap) vs balanced model
- `bias_variance_tradeoff.png`: Polynomial degrees 1, 3, 15 (underfitting → balanced → overfitting)
- `performance_degradation.png`: Out-of-time F1 decay from 0.08 (validation) to 0.48–0.58 (production)
- `concept_drift_simulation.png`: Income distribution shift ($60K → $45K mean) over 6 quarters
- `psi_trend.png`: PSI trend from 0 to 1.975 with 0.25 danger threshold
- `csi_analysis.png`: Feature-level CSI (annual income = 1.975, age = 0.011)

**Code**: `Q3_Bias_Variance_Drift.ipynb`

---


## Technical Report

A comprehensive 2–3 page technical report is available in **`report/report.qmd`** (Quarto markdown format), covering:

1. **Q1**: L1 regularization mechanism; multicollinearity comparison; production recommendation
2. **Q2**: Accuracy paradox; F1/AUCPR justification; SMOTE vs cost-sensitive empirical comparison
3. **Q3**: Bias-variance diagnosis; PSI/CSI mathematical formulation; hybrid retraining strategy
4. **Implementation Roadmap**: 4-week deployment plan with monitoring automation

**To render PDF**:
```bash
quarto render report/report.qmd --to pdf
```

---

## Key Findings

| Challenge | Solution | Evidence |
|:----------|:---------|:-------:|
| **100+ correlated features** | L1 regularization (Lasso) | 94% dimensionality reduction; stable under multicollinearity |
| **5.4% default rate** | Cost-sensitive learning + F1/AUCPR | F1 = 0.742; variance σ = 0.017 (robust) |
| **Production concept drift** | PSI/CSI monitoring + hybrid retraining | PSI breach 2–3 months before F1 collapse; early warning prevents >10% loss |

---

## Requirements

```
python >= 3.8
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
seaborn >= 0.11
scikit-learn >= 1.0
imbalanced-learn >= 0.9
statsmodels >= 0.13
quarto >= 1.3  # For rendering report.qmd
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## How to Run

1. **Jupyter Notebooks** (Interactive exploration):
   ```bash
   jupyter notebook Q1_Feature_Selection.ipynb
   jupyter notebook Q2_Class_Imbalance.ipynb
   jupyter notebook Q3_Bias_Variance_Drift.ipynb
   ```
   - Run all cells sequentially; visualizations saved to `output/` folder automatically

2. **Technical Report** (PDF generation):
   ```bash
   cd report/
   quarto render report.qmd --to pdf
   # Output: report.pdf (2–3 pages)
   ```

---
## Regulatory Compliance

- **Interpretability**: L1-penalized logistic regression is fully auditable; coefficients directly interpretable
- **Fairness**: Class-weighted loss prevents bias toward majority class
- **Monitoring**: Automated PSI/CSI alerts for model governance review
- **Documentation**: All decisions logged for regulatory audit trail

---

## References

- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- Naeem Siddiqi (2006). "Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring"
- Quarto Documentation: https://quarto.org

# hanwhafinance

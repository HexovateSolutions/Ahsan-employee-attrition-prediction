# Employee Attrition Prediction


### Project Overview
End-to-end machine learning pipeline to predict employee attrition using the
IBM HR Analytics dataset (1,470 records, 35 features). The goal is to identify
employees likely to leave so HR teams can intervene early.

---

### Streamlit App (New)
This repository now includes a lightweight Streamlit entry point to support
future deployment of the attrition prediction workflow in an HR-facing UI.

- App entry point: `app/app.py`
- Streamlit config: `.streamlit/config.toml`
- Python dependencies: `requirements.txt`

The current app includes project-appropriate placeholder sections for:
- Model input
- Prediction output
- Explainability (SHAP-ready area)

---

### Quick Start (Streamlit)
1. Create and activate a virtual environment:
   - **Linux/macOS**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app/app.py
   ```

---

### Repository Structure
```text
.
├── app/
│   └── app.py
├── .streamlit/
│   └── config.toml
├── Notebook/
│   └── employee-attrition-prediction.ipynb
├── models/
│   ├── feature_columns.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression_model.pkl
│   ├── model_metadata.json
│   ├── ohe_columns.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── requirements.txt
├── Report.pdf
└── README.md
```

---

### Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

### Key Results — Engineered Pipeline (Single Split Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.7857 | 0.3919 | **0.6170** | 0.4793 |
| Decision Tree | 0.7857 | 0.3261 | 0.3191 | 0.3226 |
| Random Forest | 0.8435 | 0.5217 | 0.2553 | 0.3429 |
| XGBoost | 0.8605 | 0.6364 | 0.2979 | 0.4058 |

> Recall is the primary metric — missing an actual leaver (false negative) costs more than a false alarm in HR context.

---

### After Threshold Tuning — Engineered Pipeline

| Model | Optimal Threshold | Tuned Recall | Tuned F1 |
|---|---|---|---|
| Logistic Regression | 0.51 | 0.6170 | 0.4915 |
| Decision Tree | 1.00 | 0.3191 | 0.3226 |
| Random Forest | **0.30** | **0.7234** | **0.5231** |
| XGBoost | 0.03 | 0.7660 | 0.4645 |

---

### ROC-AUC Scores — Engineered Pipeline

| Model | ROC-AUC |
|---|---|
| XGBoost | 0.7889 |
| Logistic Regression | 0.7875 |
| Random Forest | 0.7857 |
| Decision Tree | 0.5968 |

---

### Model Recommendation

**Without threshold tuning:** Logistic Regression (Engineered Pipeline) — highest default Recall of 0.6170, catches 29 of 47 actual leavers, fully interpretable via model coefficients.

**With threshold tuning:** Random Forest at threshold 0.30 (Engineered Pipeline) — best practical performance with Recall 0.7234 and F1 0.5231, catching ~34 of 47 actual leavers. This is the recommended production model.

---

### Techniques Used
- Exploratory Data Analysis (EDA)
- Label Encoding and One-Hot Encoding
- StandardScaler normalization
- SMOTE oversampling (applied to training data only — zero test set contamination)
- IQR-based outlier capping (MonthlyIncome, TotalWorkingYears, YearsSinceLastPromotion, NumCompaniesWorked, TrainingTimesLastYear)
- Composite feature engineering: IncomePerYear, SatisfactionIndex, PromotionLag, YearsWithManagerRatio
- 5-Fold Stratified Cross-Validation
- GridSearchCV Hyperparameter Tuning (216 combinations, 1,080 fits)
- Probability threshold optimisation
- SHAP explainability

---

### Top Findings
- **OverTime** is the single strongest predictor — SHAP value of 1.30, more than double the next feature
- **StockOptionLevel** and **MonthlyIncome** are the top financial retention signals
- Feature engineering improved Logistic Regression Recall from 0.5957 → 0.6170
- Random Forest at threshold 0.30 achieves Recall of 0.7234 after tuning — a 47 percentage point improvement
- All three top models (LR, RF, XGBoost) achieve near-identical ROC-AUC of ~0.79 — threshold tuning is the decisive factor
- Sales Representatives had the highest attrition (~40%); Research Directors the lowest (~3–5%)
- Single employees showed 25% attrition vs 10% for divorced employees

---
### Dataset
IBM HR Analytics Employee Attrition dataset
- Records: 1,470 employees | Features: 35
- Class imbalance: 83.88% No (stayed) / 16.12% Yes (left)
- Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-data-set

---

### Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, CV |
| XGBoost | Gradient boosting |
| Imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| Matplotlib, Seaborn | Visualisation |
| Kaggle Notebook (CPU) | Platform |

---



## 🚀 Live App
[Click here to view the app](https://loan-default-prediction-amzf67bcjgwziw8tkj28gp.streamlit.app)

# ðŸ’³ Loan Default Predictor

An end-to-end machine learning project to predict whether a loan applicant will default.

## ðŸ“ Project Structure

```
claude/
â”œâ”€â”€ data/
â”‚   â””â”€â”€ credit_risk_dataset.csv    # Dataset (32,581 rows, 12 features)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ preprocess.py              # Data cleaning & preprocessing
â”‚   â””â”€â”€ train.py                   # Model training & evaluation
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ best_model.pkl             # Trained XGBoost model
â”‚   â”œâ”€â”€ scaler.pkl                 # Feature scaler
â”‚   â”œâ”€â”€ label_encoders.pkl         # Categorical encoders
â”‚   â”œâ”€â”€ feature_names.pkl          # Feature column names
â”‚   â””â”€â”€ training_summary.pkl       # All model results
â”œâ”€â”€ app/
â”‚   â””â”€â”€ app.py                     # Streamlit web app
â”œâ”€â”€ notebooks/                     # (for EDA notebooks)
â””â”€â”€ requirements.txt
```

## ðŸš€ Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Run the web app
```bash
streamlit run app/app.py
```

## ðŸ“Š Model Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 84.46%   | 85.30%  |
| Random Forest       | 93.38%   | 93.22%  |
| Gradient Boosting   | 92.47%   | 92.81%  |
| **XGBoost** âœ…      | **93.61%** | **94.83%** |

## ðŸŽ¯ Target Variable
- `loan_status = 0` â†’ Loan repaid successfully
- `loan_status = 1` â†’ Loan defaulted


# 💳 Loan Default Predictor

An end-to-end machine learning project to predict whether a loan applicant will default.

## 📁 Project Structure

```
claude/
├── data/
│   └── credit_risk_dataset.csv    # Dataset (32,581 rows, 12 features)
├── src/
│   ├── preprocess.py              # Data cleaning & preprocessing
│   └── train.py                   # Model training & evaluation
├── models/
│   ├── best_model.pkl             # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoders.pkl         # Categorical encoders
│   ├── feature_names.pkl          # Feature column names
│   └── training_summary.pkl       # All model results
├── app/
│   └── app.py                     # Streamlit web app
├── notebooks/                     # (for EDA notebooks)
└── requirements.txt
```

## 🚀 Getting Started

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

## 📊 Model Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 84.46%   | 85.30%  |
| Random Forest       | 93.38%   | 93.22%  |
| Gradient Boosting   | 92.47%   | 92.81%  |
| **XGBoost** ✅      | **93.61%** | **94.83%** |

## 🎯 Target Variable
- `loan_status = 0` → Loan repaid successfully
- `loan_status = 1` → Loan defaulted

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['loan_status'].value_counts()}")

    # Drop rows with missing values (small % of data)
    df = df.dropna()

    # Encode categorical columns
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Remove outliers in age and employment length
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]

    # Features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, le_dict, X.columns.tolist()


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_and_preprocess(os.path.join(base, "data", "credit_risk_dataset.csv"))

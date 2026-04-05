import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             accuracy_score, confusion_matrix)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess


def train_and_evaluate(data_path, models_dir):
    X_train, X_test, y_train, y_test, scaler, le_dict, feature_names = \
        load_and_preprocess(data_path)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                             use_label_encoder=False, eval_metric='logloss'),
    }

    results = {}
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        results[name] = {"model": model, "accuracy": acc, "auc": auc}

        print(f"   Accuracy : {acc:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    # Pick best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    print(f"\n🏆 Best Model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(scaler,     os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(le_dict,    os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))

    # Save results summary
    summary = {k: {"accuracy": v["accuracy"], "auc": v["auc"]} for k, v in results.items()}
    joblib.dump({"best_model_name": best_name, "results": summary},
                os.path.join(models_dir, "training_summary.pkl"))

    print(f"\n✅ Model and artifacts saved to: {models_dir}")
    return best_model, scaler, le_dict, feature_names


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_evaluate(
        data_path=os.path.join(base, "data", "credit_risk_dataset.csv"),
        models_dir=os.path.join(base, "models")
    )

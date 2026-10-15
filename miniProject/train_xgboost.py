"""
Train the FINAL model (XGBoost) and save it where the API loads it from.

Usage (run from the project root — same folder as this script, data_utils.py,
and your CSV):

    pip install pandas scikit-learn xgboost joblib
    python train_xgboost.py

Output: backend/model/fire_model.pkl
"""

from pathlib import Path

import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

from data_utils import load_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Trihourly_Weather_Dataset.csv"
MODEL_OUT = ROOT / "backend" / "model" / "fire_model.pkl"


def main():
    X_train, X_test, y_train, y_test, imbalance_ratio = load_dataset(DATA_PATH)

    base_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=imbalance_ratio,  # compensates for the N/Y class imbalance
        gamma=0.5,
        max_depth=6,
        min_child_weight=3,
        n_estimators=300,
        learning_rate=0.05,
    )
    base_model.fit(X_train, y_train)

    # Calibrate so predict_proba gives well-behaved probabilities
    # (model_service.py uses predict_proba, not raw predict).
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("=== XGBoost (final model) ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1:        {f1_score(y_test, preds):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()

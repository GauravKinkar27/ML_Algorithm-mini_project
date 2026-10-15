"""
Train several OTHER classifiers purely for comparison against XGBoost.
None of these are saved as the production model — this is just to show
why XGBoost was chosen. Run train_xgboost.py separately for the real model.

Usage:
    pip install pandas scikit-learn
    python train_other_models.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from data_utils import load_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Trihourly_Weather_Dataset.csv"


def evaluate(name, model, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    }


def main():
    X_train, X_test, y_train, y_test, imbalance_ratio = load_dataset(DATA_PATH)
    class_weight = {0: 1.0, 1: imbalance_ratio}

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42), False),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42), False),
        ("SVM (RBF)", SVC(probability=True, class_weight="balanced", random_state=42), True),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=15), True),
    ]

    results = [evaluate(name, model, X_train, X_test, y_train, y_test, scale) for name, model, scale in models]

    df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
    print("\n=== Comparison (for reference only — XGBoost is the model actually deployed) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

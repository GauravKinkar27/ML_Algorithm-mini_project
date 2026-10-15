"""
Shared data loading for both training scripts.
Keeps the cleaning logic in one place so the XGBoost script and the
comparison script are guaranteed to train/test on identical data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# The 10 real weather/soil features — matches FEATURE_NAMES in
# backend/services/model_service.py. Do not add 'datetime' here: in this
# file it is a leaked severity-like value that almost perfectly separates
# the classes (N: ~0-21, Y: ~22-43) and would never be available at
# real prediction time.
FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
]


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)

    # Drop the one fully-blank trailing row some exports of this file have.
    df = df.dropna(subset=["forest_fire"] + FEATURE_COLUMNS)

    # Target: Y/N -> 1/0
    y = df["forest_fire"].map({"Y": 1, "N": 0}).astype(int)
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Class imbalance ratio (N count / Y count) — used as scale_pos_weight for
    # tree models and class_weight for linear/other models.
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    imbalance_ratio = neg / pos

    return X_train, X_test, y_train, y_test, imbalance_ratio

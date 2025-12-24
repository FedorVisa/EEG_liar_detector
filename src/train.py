import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_model(
    features_df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[CalibratedClassifierCV, StandardScaler, Dict]:
    y = features_df[label_col].values.astype(int)

    # Keep only numeric feature columns
    X_all = features_df.drop(columns=[label_col]).copy()
    numeric_cols = X_all.select_dtypes(include=["number"]).columns.tolist()
    X = X_all[numeric_cols]

    # Clean NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    # Class balance check
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    min_count = min(class_counts.values()) if class_counts else 0

    # Train/test split with safe fallback when stratify is impossible
    stratify_labels = y if min_count >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # If too few minority samples, skip CV/calibration to avoid errors
    if min_count < 2:
        base_model = RandomForestClassifier(
            random_state=random_state,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
        )
        base_model.fit(X_train_scaled, y_train)
        y_proba = base_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        metrics = {
            "note": "Calibration/grid search skipped due to minority class count < 2",
            "class_counts": class_counts,
        }
        if len(y_test) > 0 and len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["report"] = classification_report(y_test, y_pred, output_dict=True)
        return base_model, scaler, metrics

    cv_splits = min(n_splits, min_count)
    cv_splits = max(cv_splits, 2)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring="roc_auc")
    grid.fit(X_train_scaled, y_train)

    best_rf = grid.best_estimator_
    calibrated = CalibratedClassifierCV(best_rf, method="sigmoid", cv=cv)
    calibrated.fit(X_train_scaled, y_train)

    y_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "roc_auc": float(auc),
        "report": report,
        "best_params": grid.best_params_,
        "class_counts": class_counts,
    }

    return calibrated, scaler, metrics


def save_artifacts(model, scaler, feature_names, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    with open(out / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)


def load_artifacts(out_dir: str):
    out = Path(out_dir)
    model = joblib.load(out / "model.joblib")
    scaler = joblib.load(out / "scaler.joblib")
    with open(out / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

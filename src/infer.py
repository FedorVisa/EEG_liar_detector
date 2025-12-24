from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import load_bcrx, window_signal
from .features import extract_features_from_windows
from .train import load_artifacts


def predict_file(
    path: str,
    model_dir: str,
    fs: float = 500.0,
    window_sec: float = 10.0,
    step_sec: float = 2.0,
    prob_threshold: float = 0.5,
) -> Dict:
    model, scaler, feature_names = load_artifacts(model_dir)
    df = load_bcrx(path)
    windows = window_signal(df, fs=fs, window_sec=window_sec, step_sec=step_sec)
    feat_df = extract_features_from_windows(windows, fs=fs)
    # Ensure feature order and fill missing
    X = pd.DataFrame()
    for col in feature_names:
        X[col] = feat_df.get(col, 0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    preds = (proba >= prob_threshold).astype(int)

    # Aggregate metrics
    result = {
        "proba_per_window": proba.tolist(),
        "pred_per_window": preds.tolist(),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "mean_proba": float(np.mean(proba)) if len(proba) else 0.0,
        "max_proba": float(np.max(proba)) if len(proba) else 0.0,
    }

    # Pass through selected physiological metrics per window for visualization
    phys_cols = [
        "hr_mean",
        "hr_std",
        "rr_std",
        "ppg_hr_mean",
        "resp_rate_ecg",
        "resp_rate_ppg",
        "spo2_mean",
    ]
    phys_df = pd.DataFrame()
    for col in phys_cols:
        if col in feat_df.columns:
            phys_df[col] = feat_df[col]
    result["phys_metrics"] = phys_df.to_dict(orient="list") if not phys_df.empty else {}

    # Mark intervals of likely lie
    intervals: List[Tuple[float, float, float]] = []
    start_times = np.arange(0, len(windows)) * step_sec
    for i, p in enumerate(proba):
        if p >= prob_threshold:
            intervals.append((float(start_times[i]), float(start_times[i] + window_sec), float(p)))
    result["lie_intervals"] = intervals
    return result

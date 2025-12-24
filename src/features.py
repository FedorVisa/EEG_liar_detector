import logging
from typing import Dict, List

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, find_peaks

try:
    from hrvanalysis import (
        get_frequency_domain_features,
        get_time_domain_features,
        interpolate_nan_values,
        remove_ectopic_beats,
        remove_outliers,
    )

    HRV_AVAILABLE = True
except Exception:
    HRV_AVAILABLE = False

logger = logging.getLogger(__name__)


def extract_ecg_features(ecg_signal: np.ndarray, sampling_rate: float = 500) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        if "ECG_R_Peaks" in rpeaks and rpeaks["ECG_R_Peaks"].sum() > 1:
            r_peaks_indices = np.where(rpeaks["ECG_R_Peaks"] == 1)[0]
            rr_intervals = np.diff(r_peaks_indices) / sampling_rate * 1000.0
            if len(rr_intervals) > 10:
                features["hr_mean"] = 60000.0 / np.mean(rr_intervals)
                features["hr_std"] = np.std(60000.0 / rr_intervals)
                features["rr_mean"] = np.mean(rr_intervals)
                features["rr_std"] = np.std(rr_intervals)
                features["rr_rmssd"] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
                if HRV_AVAILABLE and len(rr_intervals) > 20:
                    try:
                        rr_clean = remove_outliers(rr_intervals, low_rri=300, high_rri=2000)
                        rr_clean = remove_ectopic_beats(rr_clean, method="malik")
                        rr_clean = interpolate_nan_values(rr_clean)
                        time_feat = get_time_domain_features(rr_clean)
                        for k, v in time_feat.items():
                            features[f"hrv_time_{k}"] = v
                        freq_feat = get_frequency_domain_features(rr_clean)
                        for k, v in freq_feat.items():
                            features[f"hrv_freq_{k}"] = v
                    except Exception:
                        pass
                features["ecg_mean"] = float(np.mean(ecg_signal))
                features["ecg_std"] = float(np.std(ecg_signal))
                features["ecg_min"] = float(np.min(ecg_signal))
                features["ecg_max"] = float(np.max(ecg_signal))
                features["ecg_range"] = features["ecg_max"] - features["ecg_min"]
                features["ecg_skewness"] = float(skew(ecg_signal))
                features["ecg_kurtosis"] = float(kurtosis(ecg_signal))
    except Exception as e:
        logger.warning("ECG feature extraction failed: %s", e)
    return features


def extract_ppg_features(ppg_signal: np.ndarray, spo2_signal: np.ndarray, sampling_rate: float = 500) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        features["ppg_mean"] = float(np.mean(ppg_signal))
        features["ppg_std"] = float(np.std(ppg_signal))
        features["ppg_min"] = float(np.min(ppg_signal))
        features["ppg_max"] = float(np.max(ppg_signal))
        features["ppg_range"] = features["ppg_max"] - features["ppg_min"]
        peaks, _ = find_peaks(ppg_signal, distance=sampling_rate * 0.4)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sampling_rate * 1000.0
            features["ppg_hr_mean"] = 60000.0 / np.mean(peak_intervals)
            features["ppg_hr_std"] = np.std(60000.0 / peak_intervals)
            features["ppg_interval_std"] = np.std(peak_intervals)
            peak_amplitudes = ppg_signal[peaks]
            features["ppg_amplitude_mean"] = float(np.mean(peak_amplitudes))
            features["ppg_amplitude_std"] = float(np.std(peak_amplitudes))
        features["spo2_mean"] = float(np.mean(spo2_signal))
        features["spo2_std"] = float(np.std(spo2_signal))
        features["spo2_min"] = float(np.min(spo2_signal))
        features["ppg_skewness"] = float(skew(ppg_signal))
        features["ppg_kurtosis"] = float(kurtosis(ppg_signal))
    except Exception as e:
        logger.warning("PPG feature extraction failed: %s", e)
    return features


def extract_respiratory_features(ecg_signal: np.ndarray, ppg_signal: np.ndarray, sampling_rate: float = 500) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        sos = butter(4, [0.1, 0.5], btype="band", fs=sampling_rate, output="sos")
        ecg_respiratory = filtfilt(sos[0], sos[1], ecg_signal)
        peaks_ecg, _ = find_peaks(ecg_respiratory, distance=sampling_rate * 1.5)
        if len(peaks_ecg) > 2:
            breath_intervals_ecg = np.diff(peaks_ecg) / sampling_rate
            features["resp_rate_ecg"] = 60.0 / np.mean(breath_intervals_ecg)
            features["resp_rate_std_ecg"] = np.std(60.0 / breath_intervals_ecg)
        ppg_respiratory = filtfilt(sos[0], sos[1], ppg_signal)
        peaks_ppg, _ = find_peaks(ppg_respiratory, distance=sampling_rate * 1.5)
        if len(peaks_ppg) > 2:
            breath_intervals_ppg = np.diff(peaks_ppg) / sampling_rate
            features["resp_rate_ppg"] = 60.0 / np.mean(breath_intervals_ppg)
            features["resp_rate_std_ppg"] = np.std(60.0 / breath_intervals_ppg)
    except Exception as e:
        logger.warning("Respiratory feature extraction failed: %s", e)
    return features


def extract_all_features(segment: Dict, fs: float = 500.0) -> Dict[str, float]:
    data: pd.DataFrame = segment["data"]
    features: Dict[str, float] = {}
    features["duration"] = float(segment.get("duration", len(data) / fs))
    if "label" in segment:
        features["label"] = int(segment["label"])
    # Signals
    ecg_signal = data.get("ECG", pd.Series(dtype=float)).to_numpy()
    ppg_signal = data.get("PPG", pd.Series(dtype=float)).to_numpy()
    spo2_signal = data.get("SpO2", pd.Series(dtype=float)).to_numpy()
    hr_signal = data.get("HeartRate", pd.Series(dtype=float)).to_numpy()
    features.update(extract_ecg_features(ecg_signal, sampling_rate=fs))
    features.update(extract_ppg_features(ppg_signal, spo2_signal, sampling_rate=fs))
    features.update(extract_respiratory_features(ecg_signal, ppg_signal, sampling_rate=fs))
    features["hr_monitor_mean"] = float(np.mean(hr_signal)) if hr_signal.size else np.nan
    features["hr_monitor_std"] = float(np.std(hr_signal)) if hr_signal.size else np.nan
    features["hr_monitor_min"] = float(np.min(hr_signal)) if hr_signal.size else np.nan
    features["hr_monitor_max"] = float(np.max(hr_signal)) if hr_signal.size else np.nan
    return features


def extract_features_from_windows(windows: List[pd.DataFrame], fs: float = 500.0) -> pd.DataFrame:
    rows: List[Dict] = []
    for w in windows:
        seg = {
            "data": w,
            "duration": len(w) / fs,
        }
        rows.append(extract_all_features(seg, fs=fs))
    return pd.DataFrame(rows)

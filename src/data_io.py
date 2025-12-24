import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_CHANNEL_MAP = {
    "ECG": "ECG",
    "PPG": "PPG pulse",
    "SPO2": "SpO2 pulse",
    "HR": "Heart Rate pulse",
    "TIME": "Elapsed Time",
}


def _read_csv_guess(path: Path, sep_candidates: Tuple[str, ...] = (",", ";", "\t")) -> pd.DataFrame:
    for sep in sep_candidates:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse {path} as CSV with common separators")


def _read_zipped_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(path) as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not csv_members:
                return None
            # Use the first CSV inside the archive
            with zf.open(csv_members[0]) as fh:
                df = pd.read_csv(fh)
                return df
    except Exception:
        return None


def load_bcrx(path: str, channel_map: Dict[str, str] = DEFAULT_CHANNEL_MAP) -> pd.DataFrame:
    """Attempt to load a BioRadio .bcrx file into a DataFrame.

    Tries the following fallbacks:
    1) If path looks like CSV (plain text), read directly.
    2) If it is a ZIP container with CSV inside, read the first CSV member.
    3) If path already ends with .csv, read via pandas.

    The function normalizes column names according to ``channel_map`` and adds
    ``time_seconds`` if a time column exists.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # If user already converted to CSV
    if p.suffix.lower() == ".csv":
        df = _read_csv_guess(p)
    else:
        # Try direct CSV parse (some vendors store text with .bcrx extension)
        try:
            df = _read_csv_guess(p)
        except Exception:
            df = None
        # Try zipped CSV
        if df is None:
            df = _read_zipped_csv(p)
        if df is None:
            raise ValueError(
                "Unsupported .bcrx format. Please provide a CSV export or update the loader."
            )

    df = df.copy()

    # Normalize time column
    time_col = channel_map.get("TIME")
    if time_col and time_col in df.columns:
        def parse_time(t):
            if pd.isna(t):
                return np.nan
            if isinstance(t, (int, float)):
                return float(t)
            parts = str(t).split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            return pd.to_numeric(t, errors="coerce")

        df["time_seconds"] = df[time_col].apply(parse_time)
    elif "time_seconds" not in df.columns:
        # If no time, synthesize from sampling rate later in pipeline
        df["time_seconds"] = np.arange(len(df), dtype=float)

    # Rename known channels if present
    rename_map = {}
    for key, col in channel_map.items():
        if key == "TIME":
            continue
        if col in df.columns:
            rename_map[col] = key if key not in ("PPG", "SPO2", "HR") else {
                "PPG": "PPG",
                "SPO2": "SpO2",
                "HR": "HeartRate",
            }[key]
    df = df.rename(columns=rename_map)

    return df


def load_video_markup(path: str) -> pd.DataFrame:
    """Load video markup CSV with columns Start, End, Label."""
    df = pd.read_csv(path)
    for col in ("Start", "End"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path}")

    def to_seconds(x):
        if pd.isna(x):
            return np.nan
        parts = str(x).split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return pd.to_numeric(x, errors="coerce")

    df["start_seconds"] = df["Start"].apply(to_seconds)
    df["end_seconds"] = df["End"].apply(to_seconds)
    df["duration"] = df["end_seconds"] - df["start_seconds"]

    if "Label" in df.columns:
        df["label"] = df["Label"].fillna(0).astype(int)

    return df


def align_with_markup(signal_df: pd.DataFrame, markup_df: pd.DataFrame) -> List[Dict]:
    """Slice signal into labeled segments based on markup intervals."""
    segments: List[Dict] = []
    for _, row in markup_df.iterrows():
        if pd.isna(row.get("start_seconds")) or pd.isna(row.get("end_seconds")):
            continue
        label = int(row.get("label", 0))
        mask = (signal_df["time_seconds"] >= row.start_seconds) & (
            signal_df["time_seconds"] <= row.end_seconds
        )
        seg = signal_df.loc[mask].copy()
        if seg.empty:
            continue
        segments.append(
            {
                "label": label,
                "start_time": row.start_seconds,
                "end_time": row.end_seconds,
                "duration": row.end_seconds - row.start_seconds,
                "data": seg,
            }
        )
    return segments


def window_signal(df: pd.DataFrame, fs: float, window_sec: float, step_sec: float) -> List[pd.DataFrame]:
    """Split signal into overlapping windows."""
    n = len(df)
    w = int(window_sec * fs)
    s = int(step_sec * fs)
    windows: List[pd.DataFrame] = []
    if w <= 0 or s <= 0:
        return windows
    start = 0
    while start + w <= n:
        windows.append(df.iloc[start : start + w].copy())
        start += s
    return windows


def save_metadata(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_metadata(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

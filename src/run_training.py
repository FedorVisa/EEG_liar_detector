import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Support running as a script or as a module
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import align_with_markup, load_bcrx, load_video_markup
from src.features import extract_all_features
from src.train import save_artifacts, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_participants(bioradio_dir: Path, markup_dir: Path) -> List[str]:
    signal_names = {p.stem for p in bioradio_dir.glob("*.csv")}
    markup_names = {p.stem.split("_")[0] for p in markup_dir.glob("*_FRONT.csv")}
    participants = sorted(signal_names & markup_names)
    missing_markup = signal_names - markup_names
    missing_signal = markup_names - signal_names
    if missing_markup:
        logger.warning("Нет разметки для: %s", ", ".join(sorted(missing_markup)))
    if missing_signal:
        logger.warning("Нет сигналов для: %s", ", ".join(sorted(missing_signal)))
    return participants


def build_dataset(bioradio_dir: Path, markup_dir: Path, fs: float = 500.0) -> pd.DataFrame:
    participants = discover_participants(bioradio_dir, markup_dir)
    rows: List[Dict] = []
    for name in participants:
        sig_path = bioradio_dir / f"{name}.csv"
        mark_path = markup_dir / f"{name}_FRONT.csv"
        logger.info("Participant %s: loading %s", name, sig_path.name)
        signal_df = load_bcrx(str(sig_path))
        markup_df = load_video_markup(str(mark_path))
        segments = align_with_markup(signal_df, markup_df)
        logger.info("  segments: %d", len(segments))
        for seg in segments:
            seg["participant"] = name
            feats = extract_all_features(seg, fs=fs)
            feats["participant"] = name
            rows.append(feats)
    if not rows:
        raise RuntimeError("Dataset is empty: check inputs")
    return pd.DataFrame(rows)


def main():
    base = Path(__file__).resolve().parents[1]
    bioradio_dir = base / "data" / "BioRadio-20251223T133036Z-3-001" / "BioRadio"
    markup_dir = base / "data" / "video"
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"

    logger.info("Building dataset from %s", bioradio_dir)
    features_df = build_dataset(bioradio_dir, markup_dir, fs=500.0)
    logger.info("Dataset shape: %s", features_df.shape)

    # Keep only numeric feature names for persistence
    feature_names = (
        features_df.drop(columns=["label"])
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    model, scaler, metrics = train_model(
        features_df,
        label_col="label",
        test_size=0.2,
        random_state=42,
        n_splits=5,
    )
    save_artifacts(model, scaler, feature_names, artifacts_dir)

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Training done. Artifacts in %s", artifacts_dir)


if __name__ == "__main__":
    main()

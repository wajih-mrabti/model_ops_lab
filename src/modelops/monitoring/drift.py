from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def calculate_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature: str) -> dict:
    ref_mean = float(reference_df[feature].mean())
    current_mean = float(current_df[feature].mean())
    ref_std = float(reference_df[feature].std(ddof=0) or 1.0)
    drift_score = abs(current_mean - ref_mean) / ref_std

    return {
        "feature": feature,
        "reference_mean": ref_mean,
        "current_mean": current_mean,
        "drift_score": round(drift_score, 4),
    }


def check_drift(reference_path: str | Path | None = None, current_path: str | Path | None = None):
    config = load_config()
    data_cfg = config.get("data", {})

    reference_path = Path(reference_path) if reference_path else ROOT / data_cfg.get("processed_path", "data/processed/iris_processed.csv")
    current_path = Path(current_path) if current_path else ROOT / data_cfg.get("processed_path", "data/processed/iris_processed.csv")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    drift_results = []
    for column in reference_df.columns:
        if column == "species":
            continue
        drift_results.append(calculate_feature_drift(reference_df, current_df, column))

    return drift_results


if __name__ == "__main__":
    results = check_drift()
    print(results)

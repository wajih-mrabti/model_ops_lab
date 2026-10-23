from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

PathLike = Union[str, Path]


def load_config(config_path: PathLike = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ensure_raw_data(raw_path: PathLike) -> Path:
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        return raw_path

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    df["species"] = iris.target_names[iris.target]
    df.to_csv(raw_path, index=False)
    return raw_path


def prepare_data(raw_path: PathLike | None = None, output_path: PathLike | None = None, config_path: PathLike = CONFIG_PATH) -> pd.DataFrame:
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    raw_path = Path(raw_path) if raw_path is not None else ROOT / data_cfg.get("raw_path", "data/raw/iris.csv")
    output_path = Path(output_path) if output_path is not None else ROOT / data_cfg.get("processed_path", "data/processed/iris_processed.csv")

    ensure_raw_data(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "species",
        }
    )
    if "species" not in df.columns:
        raise ValueError("Input dataset must contain a 'species' target column.")

    df = df.drop_duplicates().reset_index(drop=True)
    feature_columns = [
        col for col in data_cfg.get("feature_columns", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
        if col in df.columns
    ]
    if not feature_columns:
        feature_columns = [col for col in df.columns if col != "species"]
    df = df[feature_columns + ["species"]]
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    prepared = prepare_data()
    print(f"Prepared dataset with {prepared.shape[0]} rows and {prepared.shape[1]} columns.")

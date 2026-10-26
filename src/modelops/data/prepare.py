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
    """Load project configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def ensure_raw_data(raw_path: PathLike) -> Path:
    """Create the example Iris dataset only if the raw file is missing."""
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
    df = df.drop(columns=["target"], errors="ignore")
    df.to_csv(raw_path, index=False)

    return raw_path


def prepare_data(
    raw_path: PathLike | None = None,
    output_path: PathLike | None = None,
    config_path: PathLike = CONFIG_PATH,
) -> pd.DataFrame:
    """Clean the raw dataset and produce one correctly labelled target column."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    raw_path = (
        Path(raw_path)
        if raw_path is not None
        else ROOT / data_cfg.get("raw_path", "data/raw/iris.csv")
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else ROOT / data_cfg.get(
            "processed_path", "data/processed/iris_processed.csv"
        )
    )

    ensure_raw_data(raw_path)

    df = pd.read_csv(raw_path)

    # Standardize feature column names.
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    # Prefer the existing human-readable labels when both columns exist.
    if "species" in df.columns:
        df = df.drop(columns=["target"], errors="ignore")
    elif "target" in df.columns:
        # Convert sklearn's numeric labels to readable species names.
        label_names = {
            index: name
            for index, name in enumerate(load_iris().target_names)
        }
        numeric_target = pd.to_numeric(df["target"], errors="raise")
        df["species"] = numeric_target.map(label_names)

        if df["species"].isna().any():
            raise ValueError("The Iris dataset contains unknown target labels.")

        df = df.drop(columns=["target"])
    else:
        raise ValueError(
            "Dataset must contain either a 'species' or a 'target' column."
        )

    required_features = data_cfg.get(
        "feature_columns",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )

    missing_features = [
        column for column in required_features if column not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    df = df[required_features + ["species"]]
    df = df.reset_index(drop=True)

    if df["species"].isna().any():
        raise ValueError("The target column contains missing species labels.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    prepared = prepare_data()
    print(f"Prepared dataset: {prepared.shape[0]} rows, {prepared.shape[1]} columns")
    print("Columns:", prepared.columns.tolist())
    print("Species counts:", prepared["species"].value_counts().to_dict())

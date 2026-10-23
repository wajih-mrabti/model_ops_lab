from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def evaluate_model(config_path: str | Path = CONFIG_PATH):
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    data_file = ROOT / data_cfg.get("processed_path", "data/processed/iris_processed.csv")
    feature_columns = data_cfg.get("feature_columns", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    target_column = data_cfg.get("target_column", "species")

    df = pd.read_csv(data_file)
    X = df[feature_columns]
    y = df[target_column]

    model_path = ROOT / model_cfg.get("path", "models/model.joblib")
    model = joblib.load(model_path)
    predictions = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "classification_report": classification_report(y, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }

    metrics_path = ROOT / model_cfg.get("metrics_path", "reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Evaluation accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()

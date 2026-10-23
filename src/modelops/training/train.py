from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def train_model(config_path: str | Path = CONFIG_PATH):
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    data_file = ROOT / data_cfg.get("processed_path", "data/processed/iris_processed.csv")
    feature_columns = data_cfg.get("feature_columns", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    target_column = data_cfg.get("target_column", "species")

    df = pd.read_csv(data_file)
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(training_cfg.get("test_size", 0.2)),
        random_state=int(training_cfg.get("random_state", 42)),
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    model_path = ROOT / model_cfg.get("path", "models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = ROOT / model_cfg.get("metrics_path", "reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model trained successfully. Accuracy: {accuracy:.4f}")
    return model, accuracy


if __name__ == "__main__":
    train_model()

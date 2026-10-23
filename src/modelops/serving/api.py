from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from modelops.data.prepare import prepare_data
from modelops.training.train import train_model

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model():
    config = load_config()
    model_path = ROOT / config.get("model", {}).get("path", "models/model.joblib")
    if not model_path.exists():
        prepare_data()
        train_model()
    return joblib.load(model_path)


class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI(title="ModelOps API", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    model = load_model()
    features = pd.DataFrame(
        [[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0].tolist()
    return {
        "prediction": str(prediction),
        "probabilities": {str(label): float(value) for label, value in zip(model.classes_, probabilities)},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

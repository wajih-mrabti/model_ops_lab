from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once; never train during a prediction request."""
    config = load_config()
    model_path = ROOT / config.get("model", {}).get(
        "path", "models/model.joblib"
    )

    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Train and register a model first.",
        )

    try:
        return joblib.load(model_path)
    except (OSError, ValueError, EOFError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The model artifact could not be loaded.",
        ) from exc


class PredictionInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


app = FastAPI(title="ModelOps API", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    model = load_model()

    features = pd.DataFrame(
        [[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0].tolist()

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(label): float(value)
            for label, value in zip(model.classes_, probabilities)
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

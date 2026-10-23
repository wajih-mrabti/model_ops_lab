# ModelOps Lab

A compact MLOps project template for a machine learning lifecycle with data preparation, training, evaluation, serving, monitoring, and dashboarding.

## Project structure

- `configs/config.yaml` — project configuration
- `src/modelops/data/prepare.py` — data preprocessing and feature engineering
- `src/modelops/training/train.py` — model training
- `src/modelops/evaluation/evaluate.py` — validation and metrics
- `src/modelops/serving/api.py` — FastAPI prediction service
- `src/modelops/monitoring/drift.py` — drift detection utilities
- `dashboard/app.py` — Streamlit dashboard
- `tests/` — unit and integration tests

## Quick start

1. Create a virtual environment:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Prepare the dataset:
   - `PYTHONPATH=src python -m modelops.data.prepare`
4. Train a model:
   - `PYTHONPATH=src python -m modelops.training.train`
5. Run evaluation:
   - `PYTHONPATH=src python -m modelops.evaluation.evaluate`
6. Start the API:
   - `PYTHONPATH=src uvicorn modelops.serving.api:app --reload --host 0.0.0.0 --port 8000`
7. Start the dashboard:
   - `streamlit run dashboard/app.py`

## Docker

Use Docker Compose to run the stack together:

```bash
docker-compose up --build
```

## Notes

This lab uses the Iris dataset by default so it remains lightweight and easy to run in a development environment. The architecture is intentionally modular so it can be extended with MLflow, model registry, feature stores, and production deployment patterns.

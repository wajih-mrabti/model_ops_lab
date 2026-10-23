# Architecture Overview

This repository follows a simple MLOps lifecycle pattern:

1. Data ingestion and preparation
2. Model training and validation
3. Model serving with a REST API
4. Monitoring and drift checks
5. Dashboarding and operational visibility

## Components

### Data layer
The `data/` directory contains raw and processed datasets. The `prepare.py` module ensures a reproducible cleaning and transformation step before training.

### Training layer
The training workflow reads processed data, splits it into train/test sets, fits a scikit-learn model, and persists the trained artifact to `models/model.joblib`.

### Evaluation layer
Evaluation loads the trained model and calculates classification metrics such as accuracy, confusion matrix, and classification report.

### Serving layer
The FastAPI service exposes a `/health` endpoint and a `/predict` endpoint for model inference.

### Monitoring layer
The drift module provides a lightweight baseline comparison between reference and current data distributions.

### Dashboard layer
The Streamlit dashboard surfaces dataset summaries and operational metrics, making the project easier to explore during development.

## Recommended extension points

- Add MLflow tracking for experiment metadata and model registry integration.
- Add feature flags and deployment pipelines for staging and production.
- Add Prometheus/Grafana metrics for API and model observability.
- Add CI checks using pytest and linting.

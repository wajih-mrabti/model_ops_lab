FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

CMD ["bash", "-lc", "python -m modelops.data.prepare && python -m modelops.training.train && uvicorn modelops.serving.api:app --host 0.0.0.0 --port 8000"]

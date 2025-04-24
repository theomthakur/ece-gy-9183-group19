FROM python:3.9-slim

WORKDIR /mlflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres-mlflow:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
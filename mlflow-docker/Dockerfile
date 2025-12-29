FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow_first_exp.py .

RUN mkdir -p /mlflow/mlruns

EXPOSE 5000

CMD sh -c "\
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /mlflow/mlruns \
  --host 0.0.0.0 \
  --port 5000 & \
sleep 10 && \
python mlflow_first_exp.py && \
wait"

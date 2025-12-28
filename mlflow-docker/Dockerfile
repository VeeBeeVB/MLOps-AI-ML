FROM python:3.10-slim

#Set working directory
WORKDIR /app

#Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Create directories for MLflow
RUN mkdir -p /mlflow/mlruns

#Expose MLflow UI port
EXPOSE 5000

#Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

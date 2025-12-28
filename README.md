# MLFLow Dockerize 

**Step1: Prerequisites** (verify first)
- Docker Desktop installed and running
- Make sure Docker Desktop shows “Running”
- Use Linux containers (default)

**Check Docker works**
docker --version
docker run hello-world

If hello-world runs successfully, you’re good 👍

**Step 2: Create a Project Folder**
mlflow-docker/
│
├── Dockerfile
├── requirements.txt
└── mlruns/          (will store experiments)

- Create requirements.txt just mention "mlflow"
- Create the Dockerfile

**Docker File code**
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

**Step3: Log Experiments from Python**
pip install mlflow

**Sample Mlflow Experiment script** Save under mlflow-runs/
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("docker-test")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)

**Execute the experiment**
python mlflow_first_experiment.py

Refresh UI → experiment appears ✔









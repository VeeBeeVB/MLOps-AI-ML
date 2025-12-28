import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("docker-test")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)

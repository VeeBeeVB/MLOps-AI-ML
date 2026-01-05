from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Path where ML code is mounted

def train_wrapper():
    import sys
    # ML_CODE_PATH = "/opt/airflow/mlflow_models"
    # sys.path.append(ML_CODE_PATH)
    sys.path.insert(0, "/opt/airflow")
    # sys.path.insert(0, "/opt/airflow/mlflow_models")
    from mlflow_models.model_sample.model_sample_train import train
    train()


with DAG(
    dag_id="train_model_sample",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlflow", "model_sample"],
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_wrapper,
    )

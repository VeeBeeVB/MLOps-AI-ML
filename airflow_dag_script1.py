# Import Required libraries for Apache Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import mlflow
import mlflow.xgboost


# Add the project root to Python path so we can import mlflow_training
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_vb_gb_model import model_training

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 22, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(

    dag_id='diabetes_prediction_gb_model',
    default_args=default_args,
    description='A DAG to train and log a Gradient Boosting model for diabetes prediction using MLflow',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:           

    # Define the task to train the model
    train_model_task = PythonOperator(
        task_id='model_training',
        python_callable=model_training,
    )

    # register_to_production_task = PythonOperator(
    #     task_id='register_model_to_production', 
    #     python_callable=model_training,
    #     op_kwargs={'run_id': '{{task_instance.xcom_pull(task_ids="model_training")}}'},
    # )

    # Set task dependencies if there are multiple tasks
    train_model_task 


# Import required Libraries for this airflow run
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import mlflow.xgboost
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys

# Import the mlflow module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mlflow_vb_gb_diabetes import run_experiment

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 23),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def register_model_to_production(run_id):

    model_name = "Diabetes_GB_Regressionv_new"

    # Register the model from the run
    model_uri = f"runs:/{run_id}/Diabetes_GB_Regressionv_new"
    mlflow.register_model(model_uri, model_name)

    # Get the latest version and transition to production
    client = mlflow.tracking.MlflowClient()

    # Get the latest version of the model
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

    # Transition to production (this will overwrite any existing production model)
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )
    print(f"Model {model_name} version {latest_version.version} registered to Production stage")

# Define the DAG
with DAG(
    'mlflow_vb_gb_diabetes_dag',
    default_args=default_args,
    description='A DAG to run MLflow Gradient Boosting Diabetes experiment',
    schedule=timedelta(days=1),
    catchup=False,
) as dag:

    # Define the task to run the experiment
    run_mlflow_experiment = PythonOperator(
        task_id='run_mlflow_gb_diabetes_experiment',
        python_callable=run_experiment,
    )

    register_to_production_task = PythonOperator(
    task_id='register_to_production',
    python_callable=register_model_to_production,
    op_kwargs={'run_id': '{{ task_instance.xcom_pull(task_ids="run_mlflow_gb_diabetes_experiment") }}'},
    dag=dag,
)

    # Set task dependencies if there are multiple tasks
    # For this simple DAG, there's only one task
    run_mlflow_experiment >> register_to_production_task

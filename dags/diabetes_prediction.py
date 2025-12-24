# Import required Libraries for this airflow run
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
# import mlflow.xgboost
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys


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

def load_model(stage="Production"):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions("Diabetes_GB_Regressionv_new", stages=[stage])
    if not versions:
        raise RuntimeError("No model found in MLflow registry")

    version = versions[0].version
    uri = f"models:/Diabetes_GB_Regressionv_new/{version}"
    return mlflow.sklearn.load_model(uri), version

def predict_diabetes(**kwargs):
    model, model_version = load_model()

    dag_run = kwargs.get("dag_run")
    req = dag_run.conf if dag_run and dag_run.conf else kwargs.get("params", {})

    age = float(req.get("Age", "0.312428"))
    sex = float(req.get("Sex", "0.53536"))
    bmi = float(req.get("BMI", "0.53664"))
    bp = float(req.get("BP", "0.424131"))
    s1 = float(req.get("S1", "0.466722"))
    s2 = float(req.get("S2", "0.282959"))
    s3 = float(req.get("S3", "0.17446"))
    s4 = float(req.get("S4", "0.444654"))
    s5 = float(req.get("S5", "0.382483"))
    s6 = float(req.get("S6", "0.137767"))

    X = [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]]

    pred = model.predict(X)[0]
    
    return pred


# Define the DAG
with DAG(
    'diabetes_prediction_dag',
    default_args=default_args,
    description='A DAG to predict diabetes using Gradient Boosting Regressor',
    schedule=timedelta(days=1),
    catchup=False,
    params={
        "Age": 0.312,
        "Sex": 0.535,
        "BMI": 0.536,
        "BP": 0.424,
        "S1": 0.466,
        "S2": 0.282,
        "S3": 0.174,
        "S4": 0.444,
        "S5": 0.382,
        "S6": 0.137,
    },

) as dag:
    
    predict_task = PythonOperator(
        task_id="predict_diabetes_dag",
        python_callable=predict_diabetes,
    )

    # Set task dependencies
    predict_task
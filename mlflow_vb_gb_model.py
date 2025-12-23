# Import required libraries

import numpy as np
import pandas as pd
import os
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_diabetes

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import matplotlib.pyplot as plt

def model_training():

    # Initialize MLflow tracking and set experiment name
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Diabetes_Prediction_using_GradientBoosting")

    # Define model name and initialize MLflow client
    model_name = "Diabetes_Prediction_using_GradientBoosting"
    client = MlflowClient()

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()
    # Load dataset and split into training and testing sets
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    print(y_train)
    # Start an MLflow run and train the model
        
    with mlflow.start_run(run_name="Diabetes_Prediction_using_GradientBoosting"):
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        precision_score = precision_score(y_test, predictions, average='weighted', zero_division=1)
        recall_score = recall_score(y_test, predictions, average='weighted', zero_division=1)
        f1_score = f1_score(y_test, predictions, average='weighted', zero_division=1)

        #Hyperparameter tuning results can be logged here as well
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Save and log the plot as an artifact
        plt.figure()
        plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
        plt.title("Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig("feature_importances.png")


        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision_score', precision_score)
        mlflow.log_metric('recall_score', recall_score)
        mlflow.log_metric('f1_score', f1_score)
        mlflow.log_metric('best_n_estimators', best_params['n_estimators'])
        mlflow.log_metric('best_learning_rate', best_params['learning_rate'])
        mlflow.log_artifact("feature_importances.png")
        # mlflow.log_artifact("sample_file.html")
        mlflow.sklearn.log_model(model, input_example=X_test[:2],artifact_path='model')
        print(f"Logged model with Accuracy: {accuracy}")    

        # Get the model URI for registration
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        print(f"Model URI: {model_uri}")

    # Register the model with custom versioning
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered_model.version

    return model_name, version

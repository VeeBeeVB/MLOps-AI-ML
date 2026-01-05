import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    mlflow.set_experiment("model_sample_experiment")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train[:2],
    )

    print(f"Training complete. Accuracy={acc}")
    
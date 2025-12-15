# Import necessary libraries
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Initialize MLflow tracking and set experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_Prediction_Experiment")

# Define model name and initialize MLflow client
model_name = "Diabetes_Regression_Model"
client = MlflowClient()

# Load dataset and split into training and testing sets
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Start an MLflow run and train the model
with mlflow.start_run(run_name="AutoVersion_Run") as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Log parameters, metrics, and model to MLflow artifact store
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Get the model URI for registration
    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"Logged model with MSE: {mse:.4f}")

# Register the model with versioning
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
version = registered_model.version
print(f"Model registered as {model_name} (version {version})")

# Function to get current Production model's MSE
def get_current_prod_mse(model_name):
    try:
        prod_model = client.get_latest_versions(model_name, stages=["Production"])
        if prod_model:
            run_id = prod_model[0].run_id
            metrics = client.get_run(run_id).data.metrics
            return metrics.get("mse")
    except Exception:
        return None

current_prod_mse = get_current_prod_mse(model_name)

# Promote model based on MSE comparison, if better promote to Production
if current_prod_mse is None or mse < current_prod_mse:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"New model version {version} promoted to Production!")
else:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"New model version {version} kept in Staging (not better than Production).")

# Import Required Libraries
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_RF_Regression")

# Enable MLflow system metrics logging
mlflow.enable_system_metrics_logging()


# Load Dataset and Prepare Features and Target
data = load_diabetes()
X = data.data
y = data.target

# Print Dataset Shape
print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")

# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}, Testing target shape: {y_test.shape}")

# Start MLflow Run
with mlflow.start_run() as run:
    # Initialize and Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Model Accuracy: {accuracy}")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Log Model and Metrics to MLflow
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    # Print Run ID
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

# Register the Model
model_uri = f"runs:/{run_id}/model"
model_register = mlflow.register_model(model_uri, "Diabetes_RF_Regression")
version = model_register.version
print(f"Model registered as Diabetes_RF_Regression with version: {version}")

# Promote Model to Production Stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="Diabetes_RF_Regression",
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Model version {version} promoted to Production stage.")



# Import required Libraries
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Boston_RF_Regression")

# Enable MLflow system metrics logging
mlflow.enable_system_metrics_logging()

# Load Dataset and Prepare Features and Target
data = fetch_california_housing()
X = data.data
y = data.target 


# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}, Testing target shape: {y_test.shape}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Start MLflow Run
with mlflow.start_run() as run:
    # Initialize and Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Model Accuracy: {accuracy}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Mean Squared Error: {mse}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Plot Feature Importances
    import matplotlib.pyplot as plt 
    feature_importances = model.feature_importances_
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')  
    plt.title('Feature Importances from RandomForestRegressor')
    plt.savefig('feature_importances.png')
    plt.close()

    # Log Model and Metrics to MLflow
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact('feature_importances.png')

    # Print Run ID
    run_id = run.info.run_id
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"MLflow Run ID: {run_id}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Register the Model
model_uri = f"runs:/{run_id}/model"
model_register = mlflow.register_model(model_uri, "Boston_RF_Regression")
version = model_register.version
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Model registered as Boston_RF_Regression with version: {version}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Promote the Model to "Production" Stage based on MSE Comparison
from mlflow.tracking import MlflowClient
client = mlflow.tracking.MlflowClient()
latest_versions = client.get_latest_versions(name="Boston_RF_Regression", stages=["Production"])
if not latest_versions:
    client.transition_model_version_stage(
        name="Boston_RF_Regression",
        version=version,
        stage="Production"
    )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Model version {version} promoted to Production stage.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
else:
    prod_version = latest_versions[0]
    prod_run = client.get_run(prod_version.run_id)
    prod_mse = prod_run.data.metrics["mse"]
    if mse < prod_mse:
        client.transition_model_version_stage(
            name="Boston_RF_Regression",
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Model version {version} promoted to Production stage, replacing version {prod_version.version}.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Model version {version} not promoted. Current Production model version {prod_version.version} has lower MSE ({prod_mse}).")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Import required Libraries
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_GB_Regressionv_new")

# Load Dataset and Prepare Features and Target
data = load_diabetes()
X = data.data
y = data.target

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")  
print(data) 
print(f"X Values: {X}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   


# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
print(X_train)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}, Testing target shape: {y_test.shape}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Start MLflow Run inside a function
def run_experiment():
    with mlflow.start_run(run_name="GradientBoostingRegressor") as run:
        # Initialize and Train Model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
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

        # Log Model and Metrics to MLflow
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "Diabetes_GB_Regressionv_new")

        # Print Run ID
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

    return run_id

# Register the Model
# run_id = run_experiment()
# model_uri = f"runs:/{run_id}/model"
# model_register = mlflow.register_model(model_uri, "Diabetes_GB_Regressionv_new")
# version = model_register.version
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(f"Model registered as Diabetes_GB_Regression with version: {version}")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# # Execute the experiment
# run_id = run_experiment()
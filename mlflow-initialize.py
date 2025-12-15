# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Initialize MLflow tracking and set experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_Prediction_Experiment")

# Enable system metrics logging
mlflow.enable_system_metrics_logging()

# Load dataset and split into training and testing sets
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Start an MLflow run and train the model
with mlflow.start_run(run_name="Diabetes_Prediction"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("model_type", "LinerRegression")
    mlflow.log_metric('mse', mse)
    mlflow.sklearn.log_model(model, name="model", input_example=X_test[:2])

    print(f"Logged model with MSE: {mse}")

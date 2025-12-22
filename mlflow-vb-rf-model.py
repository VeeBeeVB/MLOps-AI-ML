# Import necessary libraries
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Initialize MLflow tracking and set experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_Prediction_RF")

# Define model name and initialize MLflow client
model_name = "Diabetes_Prediction_RF"
client = MlflowClient()

# Enable system metrics logging
mlflow.enable_system_metrics_logging()

# Load dataset and split into training and testing sets
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Start an MLflow run and train the model
with mlflow.start_run(run_name="Diabetes_Prediction_RF"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    # Save and log the plot as an artifact
    plt.figure()
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.title("Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, input_example=X_test[:2])
    print(f"Logged model with Accuracy: {accuracy}")    

    # Get the model URI for registration
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    print(f"Model URI: {model_uri}")

# Register the model with versioning
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
version = registered_model.version
print(f"Model registered as {model_name} (version {version})")




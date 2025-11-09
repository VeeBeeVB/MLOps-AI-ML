import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes_Prediction_Experiment")
mlflow.enable_system_metrics_logging()

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="Diabetes_Prediction"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_param("model_type", "LinerRegression")
    mlflow.log_metric('mse', mse)
    mlflow.sklearn.log_model(model, name="model", input_example=X_test[:2])

    print(f"Logged model with MSE: {mse}")

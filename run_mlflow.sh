#Run MLFLOW 
python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

####################################################################
#Check for MLFlow Experiments
python -m mlflow experiments search
uv run mlflow experiments search

python -m mlflow runs list --experiment-id
####################################################################
#Install uv
pip install mlflow[extras] uvicorn

####################################################################
# Run Specific Model
mlflow models prepare-env -m "runs:/d73e58b3062e4c26b909e42aa9a3add0/model"


####################################################################
#Model Serving
python -m mlflow models serve -m "runs:/d73e58b3062e4c26b909e42aa9a3add0/model" -p 5000 --no-conda
uv run mlflow models serve -m "runs:/d73e58b3062e4c26b909e42aa9a3add0/model" -p 5000 --no-conda

python -m mlflow models serve -m "runs:/ea43868a403547c69bc16ad52d177b9f/model" -p 5001 --env-manager=local


uv run req.py                                                                                  

mlflow models serve -m "models:/Diabetes_Regression_Model/Production" -h 0.0.0.0 -p 5000 --env-manager=local

####################################################################
# Install FAST API Command
pip install fastapi uvicorn pydantic requests

# To serve the model using MLflow Model Serving with FAST API
python -m mlflow models serve -m "models:/Diabetes_Regression_Model/Production" -h 0.0.0.0 -p 5000 --env-manager=local

$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
set MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
mlflow models serve -m "models:/Diabetes_Prediction_using_GradientBoosting/1" -h 0.0.0.0 -p 5001 --env-manager=local

# Run the FAST API
python -m uvicorn api:app --reload

# Ping the below URL
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs


################################### MLFLOW Ubuntu Commands ###################################
# Start MLflow server
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests
import json

app = FastAPI(
    title="MLflow Diabetes Model API",
    version="2.0",
    description="API for serving MLflow Diabetes Regression model."
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

MLFLOW_URL = "http://127.0.0.1:5001/invocations"

class ModelInput(BaseModel):
    features: list = Field(
        default=[
            [0.0380759064, 0.0506801187, 0.0616962065, 0.0218723855, -0.0442234984,
             -0.0348207628, -0.0434008456, -0.002592262, 0.0199074861, -0.0176461251]
        ],
        example=[
            [0.0380759064, 0.0506801187, 0.0616962065, 0.0218723855, -0.0442234984,
             -0.0348207628, -0.0434008456, -0.002592262, 0.0199074861, -0.0176461251]
        ],
        description="List of feature arrays. Each inner list represents one sample input."
    )

@app.post("/predict", summary="Predict diabetes progression", tags=["Prediction"])
async def predict(data: ModelInput):
    """
    Send input features to MLflow model server and return predictions.
    """
    features = data.features
    if isinstance(features[0], (int, float)):
        features = [features]

    payload = {"inputs": features}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(MLFLOW_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        prediction = response.json()
        return prediction
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting MLflow server: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from MLflow s    erver")

import requests
import json
import numpy as np
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data.data[:5].tolist() 
print(X[0])

url = "http://127.0.0.1:5001/invocations"

payload = {
    "inputs": [X[0]]
}

print(payload)

headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Response status:", response.status_code)
print("Predictions:", response.json())
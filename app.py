from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Server Crash Prediction API")

# Load trained model at startup
model = joblib.load("model/model.pkl")

# Define request body schema (VERY IMPORTANT)
class ServerMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    restart_count: int


@app.get("/")
def home():
    return {"message": "Server Crash Prediction API is running 🚀"}


@app.post("/predict")
def predict(metrics: ServerMetrics):
    
    # Convert input to DataFrame
    data = pd.DataFrame([metrics.dict()])

    # Make prediction
    prediction = model.predict(data)[0]

    return {
        "crash_prediction": int(prediction),
        "meaning": "⚠️ High Crash Risk" if prediction == 1 else "✅ Server Stable"
    }

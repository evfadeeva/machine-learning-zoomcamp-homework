import pickle

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal

import pandas as pd

# 1. Request schema

class OrderFeatures(BaseModel):
    delivery_deviation: float = Field(..., ge=0)
    median_delivery_handover_min_same_location: float = Field(..., ge=0)
    vehicle_type: Literal["BIKE", "MOPED", "CAR"]
    is_pooled: Literal[0, 1]
    has_alcohol: Literal[0, 1]
    is_new_courier: Literal[0, 1]
    is_large_order: Literal[0, 1]


class PredictionResponse(BaseModel):
    predicted_delivery_handover_min: float


# 2. Initialize API

app = FastAPI(title="delivery_handover_prediction_api")

# 3. Load model

with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


# 4. Prediction function

def predict_single(features: dict) -> float:
    """
    Runs prediction on one order (already validated by pydantic).
    """
    X = pd.DataFrame([features])   # model expects list-of-dicts
    pred = pipeline.predict(X)[0]
    return float(pred)

# 5. FastAPI route

@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderFeatures):
    pred = predict_single(order.dict())
    return PredictionResponse(predicted_delivery_handover_min=pred)


# 6. For running locally

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

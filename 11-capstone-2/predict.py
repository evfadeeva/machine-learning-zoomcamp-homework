import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field


# 1. Request schema

class ListingFeatures(BaseModel):
    host_since_months_since: float
    first_review_months_since: float
    last_review_months_since: float
    host_listings_count: int
    host_local_share: float
    distance_to_city_centre: float

    has_wifi: bool
    has_kitchen: bool
    has_dedicated_workspace: bool
    has_heating: bool
    has_air_conditioning: bool

    score_comfort: float
    score_safety: float
    score_family: float
    score_luxury: float
    score_tech: float
    amenities_count: int

    review_scores_rating: float
    description_length: int
    neighborhood_overview_length: int

    accommodates: int
    bathrooms: float
    availability_365: int
    number_of_reviews: int
    estimated_occupancy_l365d: float

    room_type: Literal[
        "Entire home/apt",
        "Private room",
        "Shared room",
        "Hotel room"
    ]

    host_response_time: Literal[
        "within an hour",
        "within a few hours",
        "within a day",
        "a few days or more"
    ]

    host_response_rate: float
    host_acceptance_rate: float

    host_is_superhost_binary: Literal[0, 1]
    host_identity_verified_binary: Literal[0, 1]
    instant_bookable_binary: Literal[0, 1]


class PredictionResponse(BaseModel):
    predicted_price_class: str


# 2. Initialize API

app = FastAPI(title="airbnb_price_class_prediction_api")

@app.get("/")
def health():
    return {"status": "ok"}


# 3. Load model

with open("model.bin", "rb") as f_in:
    artifacts = pickle.load(f_in)

model = artifacts["model"]
label_encoder = artifacts["label_encoder"]



# 4. Prediction function

def predict_single(features: dict) -> str:
    """
    Runs prediction on a single listing and decodes class label.
    """
    X = pd.DataFrame([features])

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label


# 5. FastAPI route

@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingFeatures):
    pred = predict_single(listing.dict())
    return PredictionResponse(predicted_price_class=pred)


# 6. For running locally

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


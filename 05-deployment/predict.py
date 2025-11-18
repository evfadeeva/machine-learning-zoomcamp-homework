
import pickle 

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal


# request 

class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["yes", "no", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["yes", "no", "no_internet_service"]
    onlinebackup: Literal["yes", "no", "no_internet_service"]
    deviceprotection: Literal["yes", "no", "no_internet_service"]
    techsupport: Literal["yes", "no", "no_internet_service"]
    streamingtv: Literal["yes", "no", "no_internet_service"]    
    streamingmovies: Literal["yes", "no", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal["electronic_check", "mailed_check", "bank_transfer_(automatic)", "credit_card_(automatic)"]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0)
    totalcharges: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool

# response 



from typing import Dict, Any

app = FastAPI(title="churn_prediction")

# loading the file with a model
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Customer) -> PredictionResponse: 
    prob = predict_single(customer.dict())
    return PredictionResponse(
        churn_probability = prob,
        churn =  bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
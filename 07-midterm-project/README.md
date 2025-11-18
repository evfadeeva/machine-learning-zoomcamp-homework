# Delivery Handover Time Prediction

## Overview
This project predicts the *Delivery Handover Time* — the time between a courier arriving at a customer’s address and marking an order as delivered. This metric is important for improving order assignment and ETA accuracy in a food delivery company.

The project is part of the ML Zoomcamp midterm. The dataset is a mock dataset inspired by real business logic and created with the help of AI.

---

## Project description
Predict the final leg delivery handover time (minutes) for food delivery orders. This project demonstrates a full ML workflow:
- Data preparation, EDA and feature importance in `notebook.ipynb`
- Model training and parameter tuning in `train.py`
- Model saving (`model.bin`)
- FastAPI service to predict delivery time `predict.py`
- Dockerfile to containerize the service
- Dependencies `pyproject.toml`, `uv.lock` 
- Recording of working prediction application `Delivery Handover Prediction App Recording.mov`

**NB:** This repository contains mock data (`mock_handover_data_nonlinear.csv`). The approach was developed based on this mock dataset.


## Problem Description

In real-time order assignment, it’s essential to know when a courier finishes their last delivery.  
The *last leg* — **delivery handover time** — is currently predicted using a simple city-level average.  
This project aims to replace that approach with a machine learning model using order-level and courier-level features.

### Goal
Predict continuous target variable **delivery_handover_min**.

### Type
Regression problem.

---

## Machine Learning Workflow

### Implemented in `notebook.ipynb`:
- Data cleaning and preparation  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model comparison (Linear Regression, Ridge, Lasso, Random Forest, XGBoost)  
- Hyperparameter tuning (GridSearchCV)  
- Feature importance analysis  

### Final Model
A tuned **XGBoostRegressor** wrapped in a scikit-learn pipeline.


## Local Setup

### Option 1: Using `uv` (matches Dockerfile)
1. Install `uv`: https://github.com/astral-sh/uv  
2. Sync dependencies:

```bash
uv sync --locked
````

---

### Option 2: Using Pipenv (more common)

1. Install pipenv:

```bash
pip install pipenv
```

2. Install dependencies and enter virtual environment:

```bash
pipenv install --dev
pipenv shell
```

---

### Train model

```bash
python train.py
```

This will:

* Load the dataset (`mock_handover_data_nonlinear.csv`)
* Train the XGBoost model with GridSearchCV
* Evaluate on validation and test sets
* Save the final pipeline to `model.bin`

---

### Run API

```bash
python predict.py
```

* API: `http://localhost:9696`
* Swagger docs: `http://localhost:9696/docs`

**Example request:**

```bash
curl -X POST "http://localhost:9696/predict" \
-H "Content-Type: application/json" \
-d '{
  "delivery_deviation": 3.42,
  "median_delivery_handover_min_same_location": 5.6495,
  "vehicle_type": "MOPED",
  "is_pooled": 0,
  "has_alcohol": 0,
  "is_new_courier": 0,
  "is_large_order": 0
}'
```

**Example response:**

```json
{
  "predicted_delivery_handover_min": 7.82
}
```

---

## Docker Deployment

1. Build Docker image:

```bash
docker build -t predict-delivery-handover .
```

2. Run the container:

```bash
docker run -p 9696:9696 predict-delivery-handover
```

3. Test API (same curl command as above).

---

## Notes

* Input JSON must match feature names exactly.
* Model trained on mock data; real-world performance may vary.
* Containerized via Docker with `uv` ensures reproducible environment.

---

## Dependencies

* Python 3.12
* scikit-learn
* xgboost
* pandas, numpy
* fastapi
* uvicorn
* Pipenv or `uv` for environment management



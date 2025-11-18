import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import pickle


def load_data(): 
    df = pd.read_csv('mock_handover_data_nonlinear.csv')
    # current data already doesn't have missing values, but in real life scenario it is easier to drop them
    df = df.dropna().reset_index(drop=True)

    print(f"Data loaded, data shape: {df.shape}")

    return df

def train_model(df):
    # Target
    y = df["delivery_handover_min"]

    # Feature columns
    features = [
        "delivery_deviation",
        "median_delivery_handover_min_same_location",
        "vehicle_type",
        "is_pooled",
        "has_alcohol",
        "is_new_courier",
        "is_large_order"
    ]

    X = df[features]



    # First split: Train + Val vs Test (80/20)
    X_fulltrain, X_test, y_fulltrain, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Second split: Train vs Val (60/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fulltrain, y_fulltrain, test_size=0.25, random_state=42
    )



    categorical = ["vehicle_type"]
    numeric = [
        "delivery_deviation",
        "median_delivery_handover_min_same_location"
    ]

    binary = [
        "is_pooled",
        "has_alcohol",
        "is_new_courier",
        "is_large_order"
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric), 
            ("bin", "passthrough", binary)
        ]
    )



    xgb_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse"
        ))
    ])

    xgb_params = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0]
    }

    xgb_gs = GridSearchCV(xgb_pipeline, xgb_params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    xgb_gs.fit(X_train, y_train)

    print("Model training completed. Best parameters:", xgb_gs.best_params_)

    def evaluate(model, X, y):
        pred = model.predict(X)
        return {
            "RMSE": root_mean_squared_error(y, pred),
            "MAE": mean_absolute_error(y, pred),
            "R2": r2_score(y, pred)
        }

    print('XGBoost validation data set:', evaluate(xgb_gs.best_estimator_, X_val, y_val)),

    final_model = xgb_gs.best_estimator_

    print('XGBoost test data set:', evaluate(final_model, X_test, y_test)),

    return xgb_gs


def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

    print("Model saved to", filename)

df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)
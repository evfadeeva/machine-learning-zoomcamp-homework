
# 🏠 Airbnb Price Class Prediction API

This project predicts the **price category** of an Airbnb listing (`Affordable`, `Market rate`, `High-value`) based on listing, host, location, and amenities features.

The model is trained using a **Random Forest classifier** with extensive feature engineering and is exposed via a **FastAPI** service. The application is containerized using **Docker**.

---

## 📌 Project Overview

### Problem

Given Airbnb listing data, predict which **price class** a listing belongs to:

* **Affordable**
* **Market rate**
* **High-value**

The idea is to suggest to property owners if their apartment falls under one of the categories to help with better decision-making around pricing. 

### Solution

* Extensive **feature engineering** (host, amenities, location, time, text length)
* **Random Forest** model with hyperparameter tuning
* **Pipeline** with preprocessing + model
* **FastAPI** inference service
* **Dockerized** for reproducibility and deployment

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure of the data, identify patterns, and guide feature engineering decisions.
The dataset offered 79 columns, which were closely analysed, transformed and reduced to 33 features (which are still too many to be fair).

### 📍 Location Features

* Distance to Amsterdam city centre was computed using the **Haversine formula**.
* Listings closer to the city centre were generally priced higher.

### 🛠 Amenities Analysis

* Amenities were parsed from raw text into structured features.
* The **top 50 most common amenities** were identified.
* Amenities were grouped into thematic categories:

  * Comfort
  * Safety
  * Luxury
  * Family
  * Technology
* Each listing received a **normalized score per category**, capturing amenity quality rather than raw counts.

### 📝 Text Features

* Listing description and neighborhood overview lengths were used as proxy signals for listing quality and detail.
* Longer descriptions were mildly correlated with higher price classes.
* Sentiment analysis was tested as well but it didn't bring much added value compared to simple length (though was interesting to do)

### ⭐ Reviews 

* Review scores showed a positive correlation with higher price classes.


### 🔎 Key EDA Insights

* Location and amenities are the strongest drivers of price class.
* Host-related features provide valuable secondary signals.
* Simple text-derived features can improve predictive power without NLP complexity.
* Later at the feature importance stage EDA findings were partially confirmed. The most important features were: 
    * Distance to city centre is the most important indicator
    * Number of people an apartment can fit
    * Apartment type
    * Availability and reviews

---

## 🤖 Models Tested

Multiple models were evaluated during development to balance performance, interpretability, and robustness.

### 1️⃣ Logistic Regression (Baseline)

* Used as a simple baseline classifier
* Fast to train and easy to interpret
* Performance was limited due to non-linear relationships in the data

### 2️⃣ XGBoost Classifier

* Strong performance potential
* Required careful handling of:

  * Categorical variables
  * Early stopping
  * Overfitting
* Computationally expensive during tuning
* Ultimately not selected due to complexity and long training time

### 3️⃣ Random Forest Classifier ✅ (Final Model)

* Best balance of performance and stability
* Naturally handles non-linear relationships
* Robust to outliers and feature interactions
* Allows easy inspection of feature importance
* Selected as the **final production model**

Intersting that feature importance for XGBoost was quite different from Random Forest. It is very likely that high correlation between features makes them interchangeable.

---

## 📂 Project Structure

```
.
├── train.py           # Training pipeline
├── predict.py         # FastAPI inference service
├── model.bin          # Trained model + label encoder
├── listings.csv       # Input dataset
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker image definition
├── README.md          # Project documentation
```

---

## ⚙️ Requirements

### System requirements

* Python **3.10+**
* Docker (optional, for containerized deployment)

### Python packages

All required packages are listed in `requirements.txt`.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train the model

```bash
python train.py
```

This will:

* Load and clean the data
* Perform feature engineering
* Train and evaluate the Random Forest model
* Save the trained model to `model.bin`

---

### 5️⃣ Run the prediction API locally

```bash
python predict.py
```

The API will start on:

```
http://localhost:9696
```

Swagger UI:

```
http://localhost:9696/docs
```

---

## 🔮 Making Predictions

### Endpoint

`POST /predict`

### Example request (JSON)

```json
{
  "host_since_months_since": 60,
  "first_review_months_since": 48,
  "last_review_months_since": 1,
  "host_listings_count": 3,
  "host_local_share": 1.0,
  "distance_to_city_centre": 2.1,
  "has_wifi": true,
  "has_kitchen": true,
  "has_dedicated_workspace": false,
  "has_heating": true,
  "has_air_conditioning": false,
  "score_comfort": 0.42,
  "score_safety": 0.6,
  "score_family": 0.1,
  "score_luxury": 0.3,
  "score_tech": 0.2,
  "amenities_count": 12,
  "review_scores_rating": 4.8,
  "host_location_cat": "amsterdam",
  "description_length": 320,
  "neighborhood_overview_length": 200,
  "accommodates": 2,
  "bathrooms": 1,
  "availability_365": 180,
  "number_of_reviews": 45,
  "estimated_occupancy_l365d": 0.75,
  "room_type": "Entire home/apt",
  "host_response_time": "within an hour",
  "host_response_rate": 95,
  "host_acceptance_rate": 90,
  "host_is_superhost_binary": 1,
  "host_identity_verified_binary": 1,
  "instant_bookable_binary": 1
}
```

### Example response

```json
{
  "price_class": "Market rate"
}
```

---

## 🐳 Docker Deployment

### 1️⃣ Build Docker image

```bash
docker build -t airbnb-price-predictor .
```

---

### 2️⃣ Run Docker container

```bash
docker run -p 9696:9696 airbnb-price-predictor
```

---

### 3️⃣ Test the API

Open:

```
http://localhost:9696/docs
```

Or send a request using `curl` or Postman.

---

## 🧪 Model Evaluation

The model is evaluated on:

* Training set
* Validation set
* Test set

This helps detect **overfitting** and ensure generalization.

---

## 🏁 Notes

* Missing values are dropped for simplicity
* Price outliers (> €1000) are removed
* Feature engineering follows domain logic and exploratory analysis
* Pipeline ensures reproducibility

---

## 📚 Technologies Used

* Python
* pandas / numpy
* scikit-learn
* FastAPI
* Docker

---
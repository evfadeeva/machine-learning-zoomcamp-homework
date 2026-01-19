import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#from nltk.sentiment import SentimentIntensityAnalyzer
#import nltk

from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import pickle


def load_data(): 
    df = pd.read_csv('listings.csv')
    # We are dropping missing values for simplicity
    # In real life scenario we would consider transformation as some values could be missing not at random (for example, review scores)
    df = df.dropna(subset=['price']).reset_index(drop=True)

    print(f"Data loaded, data shape: {df.shape}")

    return df

def transform_data(df):
    
    print("Transforming data...")
        # price and price_class
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df = df[df['price'] < 1000]
    df['price_class'] = pd.qcut(
    df['price'],
    q=[0, 0.25, 0.75, 1],
    labels=['Affordable', 'Market rate', 'High-value'])

        # date columns
    scrape_date = pd.to_datetime("2025-09-11")
    date_cols = ["host_since", "first_review", "last_review"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_months_since"] = ((scrape_date - df[col]).dt.days / 30).round(1)

        # host local share
    df['host_local_share'] = (
    df['host_listings_count'] / df['host_total_listings_count']
).replace([np.inf, np.nan], 1)

        # distance to city centre
    city_centre_lat = 52.373056
    city_centre_lon = 4.892778

    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the distance (km) between two points on the Earth.
        """
        R = 6371  # Earth radius in km
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        return distance

    df['distance_to_city_centre'] = df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], city_centre_lat, city_centre_lon),
        axis=1
    )

    print("Distance to city centre calculated. Transforming amenities...")

        # amenities
    total_listings = len(df)

    all_amenities = df['amenities'].dropna().str.replace(r'[\[\]"]', '', regex=True)
    amenities_list = all_amenities.str.split(',').explode().str.strip()
    amenities_count = Counter(amenities_list)
    amenities_percent = {k: round(v / total_listings * 100, 1) for k, v in amenities_count.items()}
    amenities_percent_sorted = dict(sorted(amenities_percent.items(), key=lambda x: x[1], reverse=True))
    top_50 = list(amenities_percent_sorted.keys())[:50]

    df['amenities_list'] = (
        df['amenities']
        .str.strip('[]')                # remove starting and ending brackets
        .str.replace('"', '', regex=True)  # remove quotes
        .str.split(r',\s*')               # split by comma + optional spaces
    )


    # --- Individual important amenities ---
    key_amenities = ['Wifi', 'Kitchen', 'Dedicated workspace', 'Air conditioning', 'Heating']
    for amenity in key_amenities:
        df[f'has_{amenity.lower().replace(" ", "_")}'] = df['amenities_list'].apply(lambda x: amenity in x)

    # Assign to categories
    comfort = [
        "Hair dryer", "Bed linens", "Extra pillows and blankets", "Shampoo", "Body soap",
        "Essentials", "Dishes and silverware", "Dining table", "Coffee", "Shower gel",
        "Hot water kettle", "Microwave", "Freezer", "Blender", "Toaster","Refrigerator", "Dishwasher", "Oven", 
        "Wine glasses", "Hot water"
    ]

    safety = [
        "Smoke alarm", "Carbon monoxide alarm", "Fire extinguisher", "First aid kit", "Lockbox"
    ]

    luxury = [
        "Private entrance", "Room-darkening shades", "Outdoor dining area",
        "Private patio or balcony", "Outdoor furniture", "Books and reading material",
        "Laundromat nearby", "Stove", "Bathtub", "Host greets you", "Drying rack for clothing",
        "Self check-in"
    ]

    family = ['Crib', 'High chair', 'Changing table', "Children's books and toys", 'Baby monitor', 'Pool', 'Playground']
    tech = ['Netflix', 'HBO', 'Disney+', 'Apple TV', 'Sound system', 'Bluetooth', 'Game console']

    # Ensure all top-50 amenities are included in at least one category
    included = set(comfort + safety + luxury + family + tech + key_amenities)
    for amenity in top_50:
        if amenity not in included:
            comfort.append(amenity)  # default to comfort if not obvious

    # Function to compute % score
    def compute_score(amenities, category):
        return sum(amenity in amenities for amenity in category) / len(category)

    df['score_comfort'] =df['amenities_list'].apply(lambda x: compute_score(x, comfort))
    df['score_safety'] = df['amenities_list'].apply(lambda x: compute_score(x, safety))
    df['score_luxury'] = df['amenities_list'].apply(lambda x: compute_score(x, luxury))
    df['score_family'] = df['amenities_list'].apply(lambda x: compute_score(x, family))
    df['score_tech'] = df['amenities_list'].apply(lambda x: compute_score(x, tech))


    # Compute amenities_count excluding scored amenities
    df['amenities_count'] = df['amenities_list'].apply(
        lambda x: sum(1 for a in x if a not in included)
    )

    print("Amenities transformed. Transforming the rest of the data...")
        # host location
    def map_host_location(loc):
        if pd.isna(loc):
            return "other"
        
        loc = loc.lower()
        
        if "amsterdam" in loc:
            return "amsterdam"
        elif "netherlands" in loc:
            return "netherlands"
        else:
            return "other"

    df['host_location_cat'] = df['host_location'].apply(map_host_location)

        # description and neighbourhood length
    for col in ['description', 'neighborhood_overview']:
        df[f'{col}_length'] = df[col].fillna('').str.len()

        # numeric columns 
    should_be_numeric = ['host_response_rate', 'host_acceptance_rate']
    for col in should_be_numeric:
        df[col] = (
            df[col]
            .str.replace('%', '', regex=False)  # remove %
            .astype(float)                       # convert to float
    )

        # binary 
    should_be_binary = ['host_is_superhost', 'host_has_profile_pic','host_identity_verified', 'instant_bookable']

    for col in should_be_binary:
        # If it's string 't'/'f', map to 1/0. If boolean, astype(int) works too
        df[f'{col}_binary'] = df[col].map({'t': 1, 'f': 0}).fillna(0)  # keeps existing if not 't'/'f'

    features_ready = ['host_since_months_since',
        'first_review_months_since',
        'last_review_months_since',
        'host_listings_count',
        'host_local_share',
        'distance_to_city_centre',
        'has_wifi',
        'has_kitchen',
        'has_dedicated_workspace',
        'has_heating',
        'has_air_conditioning',
        'score_comfort',
        'score_safety',
        'score_family',
        'score_luxury',
        'score_tech',
        'amenities_count',
        'review_scores_rating',
        'host_location_cat',
        'description_length',
        'neighborhood_overview_length',
        'accommodates',
        'bathrooms',
        'availability_365',
        'number_of_reviews',
        'estimated_occupancy_l365d',
        'room_type',
        'host_response_time',
        'host_response_rate',
        'host_acceptance_rate',
        'host_is_superhost_binary',
        'host_identity_verified_binary',
        'instant_bookable_binary',
        'price_class']

    df = df[features_ready].dropna().reset_index(drop=True)

    print(f"Data transformed, new shape: {df.shape}")

    return df

def train_model(df):
    print("Training model...")
    # Target
    y = df["price_class"]

    X = df.drop(columns=['price_class'])

    # Label encoding is needed for XGBoost but let's keep it 
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Converts to 0,1,2

    # --- Split ---
    # First split: Train + Val vs Test (80/20)
    X_fulltrain, X_test, y_fulltrain, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Second split: Train vs Val (60/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fulltrain, y_fulltrain, test_size=0.25, random_state=42
    )

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_nominal = ['room_type', 'host_response_time']  # OneHotEncoding

    # --- ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat_nom', OneHotEncoder(handle_unknown='ignore'), categorical_nominal)
        ]
    )

    pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])

    # Only keeping the best parameters from exploration to save time
    param_grid_rf = {
    'rf__n_estimators': 200,
    'rf__max_depth': 10,
    'rf__min_samples_split': 10,
    'rf__min_samples_leaf': 5,
    'rf__max_features': 0.5
    }

    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    # Train predictions
    y_train_pred_rf = grid_rf.predict(X_train)

    # Validation predictions
    y_val_pred_rf = grid_rf.predict(X_val)


    print("--- RANDOM FOREST ---")
    print("Model training completed. Random Forest Best Params:", grid_rf.best_params_)


    print("\n=== TRAIN RESULTS ===")
    print(classification_report(y_train, y_train_pred_rf, target_names=le.classes_))

    print("\n=== VALIDATION RESULTS ===")
    print(classification_report(y_val, y_val_pred_rf, target_names=le.classes_))

        # --- Test predictions ---
    y_test_pred_rf = grid_rf.predict(X_test)

    print("\n=== TEST RESULTS ===")
    print(classification_report(y_test, y_test_pred_rf, target_names=le.classes_))

    return grid_rf, le


def save_model(filename, model, le):
    with open(filename, 'wb') as f_out:
        pickle.dump(
        {
            'model': model,
            'label_encoder': le
        },
        f_out
    )

    print("Model saved to", filename)

df = load_data()
df = transform_data(df)
pipeline, le = train_model(df)
save_model('model.bin', pipeline, le)

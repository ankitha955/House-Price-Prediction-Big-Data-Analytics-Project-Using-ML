"""
Configuration file for House Price Prediction Project.
All paths, hyperparameters, and settings are centralized here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = r"C:\Users\Lenovo\Downloads\archive\indian_house_prices_dataset.csv"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ─── Target & ID columns ──────────────────────────────────────────────────────
TARGET_COL = "Price_INR_Lakhs"
ID_COL     = "Property_ID"

# ─── Feature groups ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "BHK", "Bathrooms", "Super_Area_SqFt", "Carpet_Area_SqFt",
    "Floor_Number", "Total_Floors", "Age_of_Property",
    "Parking", "Lift_Available", "Gated_Community",
    "Distance_to_Metro_km", "Distance_to_City_Center_km",
]

CATEGORICAL_FEATURES = [
    "City", "Locality_Type", "Property_Type", "Furnishing_Status",
]

# ─── Train / Test split ───────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─── Model hyperparameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}

RF_PARAMS = {
    "n_estimators" : 300,
    "max_depth"    : None,
    "random_state" : RANDOM_STATE,
    "n_jobs"       : -1,
}

GBR_PARAMS = {
    "n_estimators" : 300,
    "learning_rate": 0.05,
    "max_depth"    : 5,
    "random_state" : RANDOM_STATE,
}

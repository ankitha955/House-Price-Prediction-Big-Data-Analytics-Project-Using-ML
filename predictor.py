"""
Predictor module – load saved model and make predictions.
"""

import pickle
import numpy as np
import pandas as pd
import os
from config import MODEL_DIR, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def load_best_model(path: str = None):
    if path is None:
        path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[Loaded] Model from {path}")
    return model


def predict_single(
    model,
    scaler,
    feature_names: list,
    city: str           = "Mumbai",
    locality_type: str  = "Premium",
    property_type: str  = "Apartment",
    bhk: int            = 3,
    bathrooms: int      = 3,
    super_area: float   = 1500.0,
    carpet_area: float  = 1200.0,
    floor_number: int   = 5,
    total_floors: int   = 20,
    age: int            = 5,
    furnishing: str     = "Semi-Furnished",
    parking: int        = 1,
    lift: int           = 1,
    gated: int          = 1,
    dist_metro: float   = 2.0,
    dist_city: float    = 8.0,
) -> float:
    """
    Predict price for a single property.
    Returns predicted price in ₹ Lakhs.
    """

    # City encoding (simple ordinal map matching LabelEncoder order)
    city_map         = {"Ahmedabad": 0, "Bangalore": 1, "Chennai": 2,
                        "Delhi": 3, "Hyderabad": 4, "Kolkata": 5,
                        "Mumbai": 6, "Pune": 7}
    locality_map     = {"Affordable": 0, "Mid-Range": 1, "Premium": 2}
    property_map     = {"Apartment": 0, "Builder Floor": 1,
                        "Independent House": 2, "Villa": 3}
    furnishing_map   = {"Furnished": 0, "Semi-Furnished": 1, "Unfurnished": 2}

    city_enc         = city_map.get(city, 6)
    locality_enc     = locality_map.get(locality_type, 2)
    property_enc     = property_map.get(property_type, 0)
    furnishing_enc   = furnishing_map.get(furnishing, 1)

    # Engineered features
    bath_per_bhk    = bathrooms / max(bhk, 1)
    carpet_ratio    = carpet_area / max(super_area, 1)
    floor_ratio     = floor_number / max(total_floors, 1)
    area_per_bhk    = super_area / max(bhk, 1)
    amenity_score   = parking + lift + gated
    total_distance  = dist_metro + dist_city
    is_new          = int(age == 0)

    row = {
        "BHK"                       : bhk,
        "Bathrooms"                 : bathrooms,
        "Super_Area_SqFt"           : super_area,
        "Carpet_Area_SqFt"          : carpet_area,
        "Floor_Number"              : floor_number,
        "Total_Floors"              : total_floors,
        "Age_of_Property"           : age,
        "Parking"                   : parking,
        "Lift_Available"            : lift,
        "Gated_Community"           : gated,
        "Distance_to_Metro_km"      : dist_metro,
        "Distance_to_City_Center_km": dist_city,
        "City"                      : city_enc,
        "Locality_Type"             : locality_enc,
        "Property_Type"             : property_enc,
        "Furnishing_Status"         : furnishing_enc,
        "Bath_per_BHK"              : bath_per_bhk,
        "Carpet_Ratio"              : carpet_ratio,
        "Floor_Ratio"               : floor_ratio,
        "Area_per_BHK"              : area_per_bhk,
        "Amenity_Score"             : amenity_score,
        "Total_Distance"            : total_distance,
        "Is_New_Property"           : is_new,
    }

    # Align to training feature order
    X = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X)
    price    = model.predict(X_scaled)[0]
    return round(float(price), 2)

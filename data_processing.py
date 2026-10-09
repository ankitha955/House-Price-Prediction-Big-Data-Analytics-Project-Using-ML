"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import (
    DATA_PATH, TARGET_COL, ID_COL,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE
)


def load_data() -> pd.DataFrame:
    """Load raw dataset and return a cleaned DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(df.head())
    print(f"\nShape   : {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")
    print("=" * 60 + "\n")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    df = df.copy()

    # Ratio features
    df["Bath_per_BHK"]    = df["Bathrooms"] / df["BHK"].clip(lower=1)
    df["Carpet_Ratio"]    = df["Carpet_Area_SqFt"] / df["Super_Area_SqFt"].clip(lower=1)
    df["Floor_Ratio"]     = df["Floor_Number"]      / df["Total_Floors"].clip(lower=1)
    df["Area_per_BHK"]    = df["Super_Area_SqFt"]   / df["BHK"].clip(lower=1)

    # Amenity score (simple composite)
    df["Amenity_Score"]   = (
        df["Parking"] + df["Lift_Available"] + df["Gated_Community"]
    )

    # Distance composite (closer = better)
    df["Total_Distance"]  = (
        df["Distance_to_Metro_km"] + df["Distance_to_City_Center_km"]
    )

    # Property is new (age == 0)
    df["Is_New_Property"] = (df["Age_of_Property"] == 0).astype(int)

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Encode categoricals and scale numerics.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = df.copy()

    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # ── Categorical encoding ──────────────────────────────────────────────────
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Build feature matrix ──────────────────────────────────────────────────
    engineered_cols = [
        "Bath_per_BHK", "Carpet_Ratio", "Floor_Ratio",
        "Area_per_BHK", "Amenity_Score", "Total_Distance", "Is_New_Property",
    ]
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + engineered_cols
    # keep only features that exist after engineering
    all_features = [f for f in all_features if f in df.columns]

    X = df[all_features]
    y = df[TARGET_COL]

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[INFO] Train size: {X_train_scaled.shape}, Test size: {X_test_scaled.shape}")
    return (
        X_train_scaled, X_test_scaled,
        y_train.values, y_test.values,
        all_features, scaler
    )


def get_processed_data():
    """Full pipeline: load → engineer → encode/scale → return splits."""
    df = load_data()
    basic_info(df)
    df = engineer_features(df)
    return encode_and_scale(df)

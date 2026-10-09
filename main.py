"""
main.py — Entry point for the House Price Prediction project.

Usage:
    python main.py              # full pipeline
    python main.py --predict    # run sample predictions only (needs trained model)
"""

import argparse
import os
import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pickle
import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
from config      import OUTPUT_DIR, MODEL_DIR, DATA_PATH
from data_processing import load_data, engineer_features, encode_and_scale, basic_info
from model_training  import train_all_models
from visualization   import run_all_eda_plots, run_model_plots_full
from predictor       import predict_single


def full_pipeline() -> None:
    print("\n" + "=" * 60)
    print("  INDIAN HOUSE PRICE PREDICTION PIPELINE")
    print("=" * 60)

    # ── 1. Load & EDA ─────────────────────────────────────────────────────────
    df_raw = load_data()
    basic_info(df_raw)

    print("\n[Step 1] Generating EDA plots …")
    run_all_eda_plots(df_raw)

    # ── 2. Feature Engineering + Preprocessing ────────────────────────────────
    print("\n[Step 2] Feature engineering & preprocessing …")
    df_eng = engineer_features(df_raw)
    X_train, X_test, y_train, y_test, feature_names, scaler = encode_and_scale(df_eng)

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\n[Step 3] Training models …")
    results, models, best_model, best_name = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # ── 4. Model plots ────────────────────────────────────────────────────────
    print("\n[Step 4] Generating model result plots …")
    run_model_plots_full(results, models, best_model, best_name, feature_names, y_test)

    # ── 5. Save scaler & feature list ─────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"\n[Saved] Scaler → {MODEL_DIR}/scaler.pkl")
    print(f"[Saved] Features → {MODEL_DIR}/features.pkl")

    # ── 6. Sample predictions ─────────────────────────────────────────────────
    run_sample_predictions(best_model, scaler, feature_names, best_name)

    print("\n" + "=" * 60)
    print(f"  [DONE]  Pipeline complete!  Plots --> {OUTPUT_DIR}")
    print("=" * 60 + "\n")


def run_sample_predictions(model, scaler, feature_names, model_name: str) -> None:
    print(f"\n[Step 5] Sample Predictions using {model_name}")
    print("-" * 60)

    samples = [
        dict(city="Mumbai", locality_type="Premium", property_type="Apartment",
             bhk=3, bathrooms=3, super_area=1800, carpet_area=1400,
             floor_number=10, total_floors=25, age=5,
             furnishing="Furnished", parking=1, lift=1, gated=1,
             dist_metro=1.5, dist_city=7.0),
        dict(city="Bangalore", locality_type="Mid-Range", property_type="Villa",
             bhk=4, bathrooms=4, super_area=3000, carpet_area=2400,
             floor_number=0, total_floors=2, age=3,
             furnishing="Semi-Furnished", parking=2, lift=0, gated=1,
             dist_metro=5.0, dist_city=12.0),
        dict(city="Ahmedabad", locality_type="Affordable", property_type="Builder Floor",
             bhk=2, bathrooms=2, super_area=950, carpet_area=750,
             floor_number=2, total_floors=5, age=15,
             furnishing="Unfurnished", parking=0, lift=0, gated=0,
             dist_metro=8.0, dist_city=15.0),
        dict(city="Delhi", locality_type="Premium", property_type="Independent House",
             bhk=5, bathrooms=5, super_area=4000, carpet_area=3200,
             floor_number=0, total_floors=3, age=10,
             furnishing="Furnished", parking=2, lift=1, gated=1,
             dist_metro=2.0, dist_city=5.0),
    ]

    for i, s in enumerate(samples, 1):
        price = predict_single(model, scaler, feature_names, **s)
        print(
            f"  Sample {i}: {s['bhk']} BHK {s['property_type']} in {s['city']} "
            f"({s['locality_type']}) → ₹ {price} Lakhs"
        )

    print("-" * 60)


def predict_mode() -> None:
    """Load saved model + scaler and run sample predictions."""
    model_path    = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "features.pkl")

    for p in [model_path, scaler_path, features_path]:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}\nRun  python main.py  first.")
            sys.exit(1)

    with open(model_path,    "rb") as f: model         = pickle.load(f)
    with open(scaler_path,   "rb") as f: scaler        = pickle.load(f)
    with open(features_path, "rb") as f: feature_names = pickle.load(f)

    run_sample_predictions(model, scaler, feature_names, "Best Saved Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument("--predict", action="store_true",
                        help="Run predictions only (requires prior training run)")
    args = parser.parse_args()

    if args.predict:
        predict_mode()
    else:
        full_pipeline()

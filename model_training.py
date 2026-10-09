"""
Model training module – trains multiple regressors and returns results.
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor
from config import XGBOOST_PARAMS, RF_PARAMS, GBR_PARAMS, MODEL_DIR


# ─── Metric helper ────────────────────────────────────────────────────────────
def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100
    print(f"  [{name}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ─── Individual trainers ──────────────────────────────────────────────────────
def train_ridge(X_train, y_train):
    m = Ridge(alpha=10)
    m.fit(X_train, y_train)
    return m


def train_lasso(X_train, y_train):
    m = Lasso(alpha=0.1, max_iter=5000)
    m.fit(X_train, y_train)
    return m


def train_random_forest(X_train, y_train):
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X_train, y_train)
    return m


def train_gradient_boosting(X_train, y_train):
    m = GradientBoostingRegressor(**GBR_PARAMS)
    m.fit(X_train, y_train)
    return m


def train_xgboost(X_train, y_train):
    m = XGBRegressor(**XGBOOST_PARAMS, verbosity=0)
    m.fit(X_train, y_train)
    return m


# ─── Master training function ──────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all models, evaluate on test set, save best model.
    Returns: results_list, trained_models_dict, best_model
    """
    trainers = {
        "Ridge Regression"     : train_ridge,
        "Lasso Regression"     : train_lasso,
        "Random Forest"        : train_random_forest,
        "Gradient Boosting"    : train_gradient_boosting,
        "XGBoost"              : train_xgboost,
    }

    results  = []
    models   = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, trainer in trainers.items():
        print(f"\n[Training] {name} …")
        model = trainer(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(name, y_test, y_pred)
        results.append(metrics)
        models[name] = (model, y_pred)

    # ── Best model by R² ──────────────────────────────────────────────────────
    best_row   = max(results, key=lambda r: r["R2"])
    best_name  = best_row["Model"]
    best_model = models[best_name][0]

    print(f"\n[Best Model] {best_name}  →  R²={best_row['R2']:.4f}")

    # ── Save best model ────────────────────────────────────────────────────────
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[Saved]  {path}")

    return results, models, best_model, best_name

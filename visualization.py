"""
Visualization module – generates and saves all EDA & model result plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR, TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price Distribution", fontsize=14, fontweight="bold")

    sns.histplot(df[TARGET_COL], bins=50, kde=True, color="steelblue", ax=axes[0])
    axes[0].set_title("Original Scale (₹ Lakhs)")
    axes[0].set_xlabel("Price (₹ Lakhs)")

    sns.histplot(np.log1p(df[TARGET_COL]), bins=50, kde=True, color="darkorange", ax=axes[1])
    axes[1].set_title("Log-Transformed")
    axes[1].set_xlabel("log(Price + 1)")

    _save(fig, "01_price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    _save(fig, "02_correlation_heatmap.png")


def plot_categorical_vs_price(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Categorical Features vs Price", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flatten(), CATEGORICAL_FEATURES):
        order = (
            df.groupby(col)[TARGET_COL].median()
              .sort_values(ascending=False).index
        )
        sns.boxplot(data=df, x=col, y=TARGET_COL, order=order,
                    palette="Set2", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, "03_categorical_vs_price.png")


def plot_numeric_vs_price(df: pd.DataFrame) -> None:
    cols_to_plot = [c for c in NUMERIC_FEATURES if c in df.columns][:8]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Numeric Features vs Price (Scatter)", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flatten(), cols_to_plot):
        ax.scatter(df[col], df[TARGET_COL], alpha=0.3, s=10, color="steelblue")
        ax.set_xlabel(col)
        ax.set_ylabel("Price (₹ Lakhs)")

    # turn off unused subplots
    for ax in axes.flatten()[len(cols_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, "04_numeric_vs_price.png")


def plot_city_price_comparison(df: pd.DataFrame) -> None:
    city_stats = (
        df.groupby("City")[TARGET_COL]
          .agg(["mean", "median"])
          .sort_values("mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(city_stats))
    width = 0.4
    ax.bar(x - width / 2, city_stats["mean"],   width, label="Mean",   color="steelblue")
    ax.bar(x + width / 2, city_stats["median"], width, label="Median", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(city_stats.index, rotation=30)
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("Average House Price by City", fontsize=14, fontweight="bold")
    ax.legend()
    _save(fig, "05_city_price_comparison.png")


def plot_bhk_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BHK Analysis", fontsize=14, fontweight="bold")

    df["BHK"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("Count by BHK")
    axes[0].set_xlabel("BHK")

    sns.boxplot(data=df, x="BHK", y=TARGET_COL, palette="Set3", ax=axes[1])
    axes[1].set_title("Price by BHK")
    _save(fig, "06_bhk_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Result Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: list) -> None:
    df_r = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
        colors = ["gold" if v == df_r[metric].max() else "steelblue"
                  for v in df_r[metric]] if metric == "R2" else \
                 ["gold" if v == df_r[metric].min() else "steelblue"
                  for v in df_r[metric]]
        ax.barh(df_r["Model"], df_r[metric], color=colors, edgecolor="black")
        ax.set_title(metric)
        ax.set_xlabel(metric)
        for i, val in enumerate(df_r[metric]):
            ax.text(val * 0.98, i, f"{val:.3f}", va="center", ha="right",
                    fontsize=9, color="black")

    plt.tight_layout()
    _save(fig, "07_model_comparison.png")


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray,
                              model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")

    # Scatter
    mn, mx = y_test.min(), y_test.max()
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
    axes[0].set_xlabel("Actual Price (₹ Lakhs)")
    axes[0].set_ylabel("Predicted Price (₹ Lakhs)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    # Residuals
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=50, kde=True, color="darkorange", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (Actual − Predicted)")

    _save(fig, "08_actual_vs_predicted.png")


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Works for tree-based models (RF, GBR, XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        print(f"  [Skip] {model_name} has no feature_importances_")
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(20)   # top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    imp.plot(kind="barh", color="steelblue", edgecolor="black", ax=ax)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, "09_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Master call
# ─────────────────────────────────────────────────────────────────────────────
def run_all_eda_plots(df: pd.DataFrame) -> None:
    print("\n[EDA Plots]")
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_categorical_vs_price(df)
    plot_numeric_vs_price(df)
    plot_city_price_comparison(df)
    plot_bhk_distribution(df)


def run_all_model_plots(results, models, best_model, best_name, feature_names) -> None:
    print("\n[Model Plots]")
    plot_model_comparison(results)
    _, y_pred_best = models[best_name]
    y_test_best = None
    # retrieve y_test from main (passed via run_all_model_plots)
    # y_test will be passed as extra arg below
    plot_feature_importance(best_model, feature_names, best_name)


def run_model_plots_full(results, models, best_model, best_name,
                          feature_names, y_test) -> None:
    print("\n[Model Plots]")
    plot_model_comparison(results)
    _, y_pred_best = models[best_name]
    plot_actual_vs_predicted(y_test, y_pred_best, best_name)
    plot_feature_importance(best_model, feature_names, best_name)

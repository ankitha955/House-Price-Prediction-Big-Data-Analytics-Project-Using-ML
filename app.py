"""
app.py  —  Streamlit Web App for Indian House Price Prediction
Run with:  streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "indian_house_prices_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEAT_PATH   = os.path.join(BASE_DIR, "models", "features.pkl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH,  "rb") as f: model    = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler   = pickle.load(f)
    with open(FEAT_PATH,   "rb") as f: features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_price(model, scaler, feature_names, inputs: dict) -> float:
    city_map       = {"Ahmedabad":0,"Bangalore":1,"Chennai":2,
                      "Delhi":3,"Hyderabad":4,"Kolkata":5,
                      "Mumbai":6,"Pune":7}
    locality_map   = {"Affordable":0,"Mid-Range":1,"Premium":2}
    property_map   = {"Apartment":0,"Builder Floor":1,
                      "Independent House":2,"Villa":3}
    furnishing_map = {"Furnished":0,"Semi-Furnished":1,"Unfurnished":2}

    sa  = inputs["super_area"]
    ca  = inputs["carpet_area"]
    bhk = inputs["bhk"]
    tf  = max(inputs["total_floors"], 1)
    fn  = inputs["floor_number"]

    row = {
        "BHK"                       : bhk,
        "Bathrooms"                 : inputs["bathrooms"],
        "Super_Area_SqFt"           : sa,
        "Carpet_Area_SqFt"          : ca,
        "Floor_Number"              : fn,
        "Total_Floors"              : tf,
        "Age_of_Property"           : inputs["age"],
        "Parking"                   : inputs["parking"],
        "Lift_Available"            : inputs["lift"],
        "Gated_Community"           : inputs["gated"],
        "Distance_to_Metro_km"      : inputs["dist_metro"],
        "Distance_to_City_Center_km": inputs["dist_city"],
        "City"                      : city_map.get(inputs["city"], 6),
        "Locality_Type"             : locality_map.get(inputs["locality"], 2),
        "Property_Type"             : property_map.get(inputs["prop_type"], 0),
        "Furnishing_Status"         : furnishing_map.get(inputs["furnishing"], 1),
        "Bath_per_BHK"              : inputs["bathrooms"] / max(bhk, 1),
        "Carpet_Ratio"              : ca / max(sa, 1),
        "Floor_Ratio"               : fn / tf,
        "Area_per_BHK"              : sa / max(bhk, 1),
        "Amenity_Score"             : inputs["parking"] + inputs["lift"] + inputs["gated"],
        "Total_Distance"            : inputs["dist_metro"] + inputs["dist_city"],
        "Is_New_Property"           : int(inputs["age"] == 0),
    }
    X = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X)
    return round(float(model.predict(X_scaled)[0]), 2)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px; padding: 20px; text-align: center; color: white;
    }
    .metric-card h2 { font-size: 2.2rem; margin: 0; color: #ffd700; }
    .metric-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    .price-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px; padding: 24px; text-align: center;
        box-shadow: 0 4px 20px rgba(56,239,125,0.35);
    }
    .price-badge h1 { color: white; font-size: 2.8rem; margin: 0; }
    .price-badge p  { color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_house_icon.svg/120px-Simple_house_icon.svg.png",
    width=80
)
st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("*Indian Real Estate · ML Powered*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏡 Predict Price", "📊 Data Explorer", "📈 Model Insights", "ℹ️ About"],
    label_visibility="collapsed"
)
st.sidebar.divider()
st.sidebar.caption("Dataset: 5,500 Indian properties · Model: XGBoost")

# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, feature_names = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False

df = load_data()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 · PREDICT PRICE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏡 Predict Price":
    st.title("🏠 Indian House Price Predictor")
    st.markdown("Fill in the property details below to get an **AI-powered price estimate**.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location")
        city     = st.selectbox("City", ["Mumbai","Delhi","Bangalore","Chennai",
                                          "Hyderabad","Kolkata","Pune","Ahmedabad"])
        locality = st.selectbox("Locality Type", ["Premium","Mid-Range","Affordable"])
        prop_type= st.selectbox("Property Type", ["Apartment","Villa",
                                                    "Independent House","Builder Floor"])

    with col2:
        st.subheader("🏗️ Property Details")
        bhk        = st.slider("BHK",            1, 8, 3)
        bathrooms  = st.slider("Bathrooms",       1, 8, 3)
        super_area = st.number_input("Super Area (sq.ft.)", 300, 10000, 1500, step=50)
        carpet_area= st.number_input("Carpet Area (sq.ft.)", 200, 8000,
                                      int(super_area * 0.78), step=50)
        age        = st.slider("Age of Property (years)", 0, 50, 5)
        floor_number = st.slider("Floor Number", 0, 50, 5)
        total_floors = st.slider("Total Floors", 1, 60, 20)

    with col3:
        st.subheader("✨ Amenities & Distance")
        furnishing = st.selectbox("Furnishing", ["Furnished","Semi-Furnished","Unfurnished"])
        parking    = st.checkbox("Parking Available",  value=True)
        lift       = st.checkbox("Lift Available",     value=True)
        gated      = st.checkbox("Gated Community",    value=True)
        st.write("")
        dist_metro = st.slider("Distance to Metro (km)", 0.1, 20.0, 2.0, 0.1)
        dist_city  = st.slider("Distance to City Center (km)", 0.5, 30.0, 8.0, 0.5)

    st.divider()

    if st.button("🔮  Predict Price", use_container_width=True, type="primary"):
        if not model_ready:
            st.error("Model not found! Run `python main.py` first to train.")
        else:
            inputs = dict(
                city=city, locality=locality, prop_type=prop_type,
                bhk=bhk, bathrooms=bathrooms,
                super_area=float(super_area), carpet_area=float(carpet_area),
                floor_number=floor_number, total_floors=total_floors,
                age=age, furnishing=furnishing,
                parking=int(parking), lift=int(lift), gated=int(gated),
                dist_metro=dist_metro, dist_city=dist_city,
            )
            with st.spinner("Calculating…"):
                price = predict_price(model, scaler, feature_names, inputs)

            # ── Result display ────────────────────────────────────────────────
            r1, r2, r3 = st.columns([1, 2, 1])
            with r2:
                st.markdown(f"""
                <div class="price-badge">
                    <p>Estimated Price</p>
                    <h1>₹ {price:,.2f} Lakhs</h1>
                    <p>≈ ₹ {price * 100_000:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            m1, m2, m3, m4 = st.columns(4)
            price_per_sqft = round(price * 100_000 / max(super_area, 1))
            m1.metric("Price / sq.ft.", f"₹ {price_per_sqft:,}")
            m2.metric("BHK",           f"{bhk} BHK")
            m3.metric("Super Area",    f"{super_area:,} sq.ft.")
            m4.metric("City",          city)

            # ── Comparable properties from dataset ────────────────────────────
            st.divider()
            st.subheader("🔍 Similar Properties in Dataset")
            similar = df[
                (df["City"] == city) &
                (df["BHK"] == bhk) &
                (df["Locality_Type"] == locality)
            ].head(5)
            if similar.empty:
                similar = df[(df["City"] == city) & (df["BHK"] == bhk)].head(5)
            if not similar.empty:
                st.dataframe(
                    similar[["City","Locality_Type","Property_Type","BHK","Bathrooms",
                              "Super_Area_SqFt","Age_of_Property","Furnishing_Status",
                              "Price_INR_Lakhs"]].reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.info("No similar properties found in the dataset for these filters.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 · DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown(f"Dataset: **{len(df):,} properties** across Indian cities")
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Properties",  f"{len(df):,}")
    k2.metric("Avg Price",         f"₹ {df['Price_INR_Lakhs'].mean():.1f} L")
    k3.metric("Median Price",      f"₹ {df['Price_INR_Lakhs'].median():.1f} L")
    k4.metric("Price Range",       f"₹ {df['Price_INR_Lakhs'].min():.0f} – {df['Price_INR_Lakhs'].max():.0f} L")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ City Analysis", "🏗️ Property Types", "📐 Area vs Price", "📋 Raw Data"]
    )

    # ── Tab 1: City ───────────────────────────────────────────────────────────
    with tab1:
        city_stats = (
            df.groupby("City")["Price_INR_Lakhs"]
              .agg(["mean","median","count"])
              .rename(columns={"mean":"Avg Price","median":"Median Price","count":"Properties"})
              .sort_values("Avg Price", ascending=False)
        )
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Average Price by City (₹ Lakhs)")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(city_stats)))
            ax.barh(city_stats.index, city_stats["Avg Price"], color=colors, edgecolor="white")
            ax.set_xlabel("Price (₹ Lakhs)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_c2:
            st.subheader("City-wise Stats")
            st.dataframe(city_stats.style.format({
                "Avg Price": "₹{:.1f}L",
                "Median Price": "₹{:.1f}L",
                "Properties": "{:,}"
            }), use_container_width=True)

        st.subheader("Price Distribution by City")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        city_order = df.groupby("City")["Price_INR_Lakhs"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="City", y="Price_INR_Lakhs", order=city_order,
                    hue="City", legend=False, palette="Set2", ax=ax2)
        ax2.set_xlabel("")
        ax2.set_ylabel("Price (₹ Lakhs)")
        ax2.set_ylim(0, df["Price_INR_Lakhs"].quantile(0.98))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tab 2: Property Types ─────────────────────────────────────────────────
    with tab2:
        p1, p2 = st.columns(2)

        with p1:
            st.subheader("Property Type Count")
            fig, ax = plt.subplots(figsize=(6, 5))
            counts = df["Property_Type"].value_counts()
            wedge_props = {"edgecolor": "white", "linewidth": 2}
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=["#2196F3","#FF9800","#4CAF50","#9C27B0"],
                   wedgeprops=wedge_props, startangle=90)
            ax.set_title("Property Type Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with p2:
            st.subheader("Price by Furnishing Status")
            fig, ax = plt.subplots(figsize=(6, 5))
            furn_order = df.groupby("Furnishing_Status")["Price_INR_Lakhs"].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x="Furnishing_Status", y="Price_INR_Lakhs",
                        order=furn_order, hue="Furnishing_Status", legend=False,
                        palette=["#4CAF50","#FF9800","#F44336"], ax=ax)
            ax.set_ylim(0, df["Price_INR_Lakhs"].quantile(0.97))
            ax.set_xlabel("")
            ax.set_ylabel("Price (₹ Lakhs)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("BHK vs Price")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        bhk_order = sorted(df["BHK"].unique())
        sns.boxplot(data=df, x="BHK", y="Price_INR_Lakhs", order=bhk_order,
                    hue="BHK", legend=False, palette="viridis", ax=ax3)
        ax3.set_ylim(0, df["Price_INR_Lakhs"].quantile(0.97))
        ax3.set_ylabel("Price (₹ Lakhs)")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── Tab 3: Area vs Price ──────────────────────────────────────────────────
    with tab3:
        st.subheader("Super Area vs Price")
        city_filter = st.multiselect("Filter by City", df["City"].unique().tolist(),
                                      default=df["City"].unique().tolist())
        filtered = df[df["City"].isin(city_filter)]

        fig, ax = plt.subplots(figsize=(12, 6))
        cities_in_filter = filtered["City"].unique()
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(cities_in_filter)))
        for city_name, color in zip(cities_in_filter, colors_map):
            mask = filtered["City"] == city_name
            ax.scatter(filtered.loc[mask, "Super_Area_SqFt"],
                       filtered.loc[mask, "Price_INR_Lakhs"],
                       alpha=0.45, s=15, label=city_name, color=color)
        ax.set_xlabel("Super Area (sq.ft.)")
        ax.set_ylabel("Price (₹ Lakhs)")
        ax.set_ylim(0, filtered["Price_INR_Lakhs"].quantile(0.98))
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Correlation bar
        st.subheader("Correlation with Price")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()["Price_INR_Lakhs"].drop("Price_INR_Lakhs").sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors2 = ["#F44336" if v < 0 else "#2196F3" for v in corr]
        ax2.barh(corr.index, corr.values, color=colors2, edgecolor="white")
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Pearson Correlation")
        ax2.set_title("Feature Correlation with Price")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tab 4: Raw Data ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("Raw Dataset")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            city_sel = st.multiselect("Filter City", df["City"].unique(), key="raw_city")
        with col_f2:
            bhk_sel  = st.multiselect("Filter BHK",  sorted(df["BHK"].unique()), key="raw_bhk")

        raw = df.copy()
        if city_sel: raw = raw[raw["City"].isin(city_sel)]
        if bhk_sel:  raw = raw[raw["BHK"].isin(bhk_sel)]
        st.dataframe(raw.reset_index(drop=True), use_container_width=True, height=450)
        st.caption(f"Showing {len(raw):,} of {len(df):,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 · MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")
    st.divider()

    results = [
        {"Model":"XGBoost",          "R²":0.757,"MAE":32.97,"RMSE":55.65,"MAPE":32.10},
        {"Model":"Gradient Boosting","R²":0.753,"MAE":32.44,"RMSE":56.01,"MAPE":31.27},
        {"Model":"Random Forest",    "R²":0.734,"MAE":33.27,"RMSE":58.13,"MAPE":32.28},
        {"Model":"Lasso Regression", "R²":0.455,"MAE":55.30,"RMSE":83.25,"MAPE":65.89},
        {"Model":"Ridge Regression", "R²":0.455,"MAE":55.35,"RMSE":83.29,"MAPE":66.00},
    ]
    df_r = pd.DataFrame(results)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Performance Metrics — All Models")
    st.dataframe(
        df_r.style
            .highlight_max(subset=["R²"], color="#d4edda")
            .highlight_min(subset=["MAE","RMSE","MAPE"], color="#d4edda")
            .format({"R²":"{:.3f}","MAE":"{:.2f}","RMSE":"{:.2f}","MAPE":"{:.2f}%"}),
        use_container_width=True
    )
    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    mc1, mc2 = st.columns(2)
    with mc1:
        st.subheader("R² Score (higher is better)")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_r = ["gold" if m == "XGBoost" else "#2196F3" for m in df_r["Model"]]
        ax.barh(df_r["Model"], df_r["R²"], color=colors_r, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        for i, v in enumerate(df_r["R²"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with mc2:
        st.subheader("MAE — Mean Absolute Error (lower is better)")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        colors_m = ["gold" if m == "Gradient Boosting" else "#FF6B6B" for m in df_r["Model"]]
        ax2.barh(df_r["Model"], df_r["MAE"], color=colors_m, edgecolor="white")
        ax2.invert_yaxis()
        for i, v in enumerate(df_r["MAE"]):
            ax2.text(v + 0.3, i, f"{v:.2f}", va="center")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Pre-generated plots ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Saved Output Plots")
    plot_files = sorted([
        f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")
    ]) if os.path.exists(OUTPUT_DIR) else []

    if plot_files:
        plot_names = {
            "01_price_distribution.png"  : "Price Distribution",
            "02_correlation_heatmap.png" : "Correlation Heatmap",
            "03_categorical_vs_price.png": "Categorical vs Price",
            "04_numeric_vs_price.png"    : "Numeric vs Price",
            "05_city_price_comparison.png":"City Price Comparison",
            "06_bhk_analysis.png"        : "BHK Analysis",
            "07_model_comparison.png"    : "Model Comparison",
            "08_actual_vs_predicted.png" : "Actual vs Predicted",
            "09_feature_importance.png"  : "Feature Importance",
        }
        selected = st.selectbox(
            "Choose a plot",
            plot_files,
            format_func=lambda f: plot_names.get(f, f)
        )
        img = Image.open(os.path.join(OUTPUT_DIR, selected))
        st.image(img, use_container_width=True)
    else:
        st.info("No plots found. Run `python main.py` to generate them.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 · ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.divider()

    st.markdown("""
    ## 🏠 Indian House Price Prediction

    This is a complete **end-to-end Machine Learning project** that predicts house
    prices across major Indian cities using the *Indian House Prices Dataset* from Kaggle.

    ---

    ### 📊 Dataset
    | Field | Details |
    |---|---|
    | **Total Records** | 5,500 properties |
    | **Cities** | Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune |
    | **Target** | `Price_INR_Lakhs` |
    | **Features** | 17 raw + 7 engineered = 24 total |

    ---

    ### 🤖 Models Used
    | Model | R² Score |
    |---|---|
    | **XGBoost** ⭐ (Best) | 0.757 |
    | Gradient Boosting | 0.753 |
    | Random Forest | 0.734 |
    | Lasso Regression | 0.455 |
    | Ridge Regression | 0.455 |

    ---

    ### 🔑 Top Price Drivers
    1. **Locality Type** — Premium vs Affordable has the biggest impact
    2. **City** — Mumbai & Delhi command highest prices
    3. **Super Area (sq.ft.)** — Larger homes cost more
    4. **Age of Property** — Newer properties are priced higher
    5. **Distance to City Center** — Closer = more expensive

    ---

    ### 🛠️ Tech Stack
    - **Python 3.14** · pandas · NumPy
    - **scikit-learn** · XGBoost
    - **Streamlit** · Matplotlib · Seaborn
    """)

# 🏠 Indian House Price Prediction

A complete end-to-end Machine Learning project for predicting house prices across major Indian cities using the **Indian House Prices Dataset** from Kaggle.

---

## 📂 Project Structure

```
house_price_prediction/
├── config.py           # Central config (paths, features, hyperparameters)
├── data_processing.py  # Data loading, feature engineering, encoding & scaling
├── model_training.py   # Train 5 ML models & evaluate
├── visualization.py    # EDA & model result plots
├── predictor.py        # Load saved model & predict single properties
├── main.py             # Entry point – full pipeline
├── requirements.txt    # Python dependencies
├── outputs/            # Generated plots (auto-created)
└── models/             # Saved model + scaler (auto-created)
```

---

## 📊 Dataset

| Field | Details |
|---|---|
| **Source** | [Kaggle – Indian House Prices Dataset](https://www.kaggle.com/) |
| **File** | `indian_house_prices_dataset.csv` |
| **Rows** | ~5,501 |
| **Target** | `Price_INR_Lakhs` |

### Features Used

| Category | Features |
|---|---|
| **Numeric** | BHK, Bathrooms, Super Area, Carpet Area, Floor, Total Floors, Age, Parking, Lift, Gated Community, Distance to Metro/City |
| **Categorical** | City, Locality Type, Property Type, Furnishing Status |
| **Engineered** | Bath/BHK ratio, Carpet Ratio, Floor Ratio, Area/BHK, Amenity Score, Total Distance, Is New Property |

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Ridge Regression | Baseline linear model |
| Lasso Regression | Feature-selecting linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting regressor |
| **XGBoost** | Usually best performer |

---

## 📈 Output Plots

| # | Plot | Description |
|---|---|---|
| 01 | Price Distribution | Original + log-transformed |
| 02 | Correlation Heatmap | Numeric feature correlations |
| 03 | Categorical vs Price | Box plots per category |
| 04 | Numeric vs Price | Scatter plots |
| 05 | City Price Comparison | Mean & median by city |
| 06 | BHK Analysis | Count & price distribution |
| 07 | Model Comparison | R², MAE, RMSE bar charts |
| 08 | Actual vs Predicted | Scatter + residual histogram |
| 09 | Feature Importance | Top 20 features (best model) |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

### Full Pipeline
```bash
python main.py
```

This will:
1. Load & explore the dataset  
2. Generate all EDA plots → `outputs/`  
3. Engineer features & preprocess data  
4. Train 5 models and evaluate on test set  
5. Save the best model → `models/best_model.pkl`  
6. Generate model result plots  
7. Print sample predictions  

### Prediction Only (after training)
```bash
python main.py --predict
```

---

## 🎯 Sample Predictions

After training, the model will predict prices like:

```
Sample 1: 3 BHK Apartment in Mumbai (Premium)      → ₹ ~230 Lakhs
Sample 2: 4 BHK Villa in Bangalore (Mid-Range)      → ₹ ~185 Lakhs
Sample 3: 2 BHK Builder Floor in Ahmedabad (Affordable) → ₹ ~45 Lakhs
Sample 4: 5 BHK Independent House in Delhi (Premium) → ₹ ~310 Lakhs
```

---

## 🛠️ Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (₹ Lakhs) |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of determination (1.0 = perfect) |
| **MAPE** | Mean Absolute Percentage Error |

---

## 📦 Requirements

```
pandas>=1.5
scikit-learn>=1.1
matplotlib>=3.5
seaborn>=0.12
xgboost>=1.7
numpy>=1.23
```

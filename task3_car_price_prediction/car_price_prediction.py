"""
Car Price Prediction with Machine Learning
===========================================
Internship Project - Task 3
CodeAlpha - Predicting Used Car Selling Prices

Features: Brand, Year, Present Price, KMs Driven,
          Fuel Type, Selling Type, Transmission, Owner
Target  : Selling_Price (in lakhs ₹)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# ─── 1. LOAD & EXPLORE ───────────────────────────────────────────────────────

print("=" * 60)
print("   CAR PRICE PREDICTION — MACHINE LEARNING")
print("=" * 60)

df = pd.read_csv("car_data.csv")
print(f"\n📊 Dataset: {df.shape[0]} cars × {df.shape[1]} features")
print(f"   Price range: ₹{df['Selling_Price'].min():.1f}L – ₹{df['Selling_Price'].max():.1f}L")
print(f"   Year range : {df['Year'].min()} – {df['Year'].max()}")

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────

# Car age is more meaningful than year
df["Car_Age"] = 2024 - df["Year"]

# Price depreciation ratio
df["Price_Ratio"] = df["Selling_Price"] / df["Present_Price"]

# Extract brand from car name (first word)
df["Brand"] = df["Car_Name"].str.split().str[0].str.lower()

# Keep top 10 brands, group rest as "other"
top_brands = df["Brand"].value_counts().head(10).index
df["Brand"] = df["Brand"].apply(lambda x: x if x in top_brands else "other")

print(f"\n🔧 Feature Engineering:")
print(f"   Added: Car_Age, Price_Ratio, Brand")
print(f"   Top brands: {', '.join(top_brands[:5])}...")

# ─── 3. PREPROCESSING ────────────────────────────────────────────────────────

df_model = df.copy()

# Encode categorical features
le = LabelEncoder()
for col in ["Fuel_Type", "Selling_type", "Transmission", "Brand"]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Features and target
features = ["Car_Age", "Present_Price", "Driven_kms", "Fuel_Type",
            "Selling_type", "Transmission", "Owner", "Brand"]
X = df_model[features]
y = df_model["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ─── 4. TRAIN MULTIPLE MODELS ────────────────────────────────────────────────

models = {
    "Linear Regression":      LinearRegression(),
    "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n🤖 Training Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "MAE":    mean_absolute_error(y_test, y_pred),
        "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":     r2_score(y_test, y_pred),
    }
    print(f"   {name:<25} R² = {results[name]['R2']:.4f} | "
          f"MAE = ₹{results[name]['MAE']:.2f}L | "
          f"RMSE = ₹{results[name]['RMSE']:.2f}L")

# Best model
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} (R² = {best['R2']:.4f})")

# ─── 5. VISUALISATIONS ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Car Price Prediction — Model Analysis", fontsize=16, fontweight="bold")

# 5a. Actual vs Predicted (best model)
ax = axes[0, 0]
ax.scatter(y_test, best["y_pred"], alpha=0.6, color="#2980B9", edgecolors="white", lw=0.3, s=60)
lims = [min(y_test.min(), best["y_pred"].min()), max(y_test.max(), best["y_pred"].max())]
ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
ax.set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
ax.set_xlabel("Actual Price (₹ Lakhs)")
ax.set_ylabel("Predicted Price (₹ Lakhs)")
ax.legend()
ax.text(0.05, 0.92, f"R² = {best['R2']:.4f}", transform=ax.transAxes,
        fontsize=11, color="#2C3E50", fontweight="bold")

# 5b. Feature Importance (Random Forest)
ax = axes[0, 1]
rf_model = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=features).sort_values()
colors    = sns.color_palette("viridis", len(feat_imp))
feat_imp.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")
for bar, val in zip(ax.patches, feat_imp):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)

# 5c. Model Comparison
ax = axes[0, 2]
model_names = list(results.keys())
r2_scores   = [results[m]["R2"]  for m in model_names]
bar_colors  = ["#E74C3C" if m == best_name else "#95A5A6" for m in model_names]
bars = ax.bar(model_names, r2_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
ax.set_title("Model Comparison (R² Score)", fontweight="bold")
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", rotation=15)

# 5d. Price distribution by fuel type
ax = axes[1, 0]
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", palette="Set2", ax=ax)
ax.set_title("Selling Price by Fuel Type", fontweight="bold")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Selling Price (₹ Lakhs)")

# 5e. Price vs Car Age
ax = axes[1, 1]
for fuel, color in zip(["Petrol", "Diesel", "CNG"], ["#E74C3C", "#2980B9", "#27AE60"]):
    mask = df["Fuel_Type"] == fuel
    ax.scatter(df.loc[mask, "Car_Age"], df.loc[mask, "Selling_Price"],
               alpha=0.5, color=color, label=fuel, s=40, edgecolors="white", lw=0.3)
ax.set_title("Car Age vs Selling Price", fontweight="bold")
ax.set_xlabel("Car Age (years)")
ax.set_ylabel("Selling Price (₹ Lakhs)")
ax.legend(fontsize=9)

# 5f. Residual plot (best model)
ax = axes[1, 2]
residuals = y_test - best["y_pred"]
ax.scatter(best["y_pred"], residuals, alpha=0.6, color="#8E44AD",
           edgecolors="white", lw=0.3, s=60)
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_title(f"Residual Plot — {best_name}", fontweight="bold")
ax.set_xlabel("Predicted Price (₹ Lakhs)")
ax.set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("car_price_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n💾 Saved: car_price_analysis.png")

# ─── 6. SAMPLE PREDICTION ────────────────────────────────────────────────────

print("\n🔮 Sample Prediction (Random Forest):")
sample = pd.DataFrame([{
    "Car_Age": 4, "Present_Price": 8.5, "Driven_kms": 35000,
    "Fuel_Type": 1, "Selling_type": 0, "Transmission": 0,
    "Owner": 0, "Brand": 5,
}])
pred_price = results["Random Forest"]["model"].predict(sample)[0]
print(f"   Car: 4-year-old, ₹8.5L present price, 35k km driven")
print(f"   Predicted Selling Price: ₹{pred_price:.2f} Lakhs")

print(f"\n{'='*60}")
print("   SUMMARY")
print(f"{'='*60}")
print(f"   Best Model : {best_name}")
print(f"   R² Score   : {best['R2']:.4f}  ({best['R2']*100:.1f}% variance explained)")
print(f"   MAE        : ₹{best['MAE']:.2f} Lakhs avg error")
print(f"   RMSE       : ₹{best['RMSE']:.2f} Lakhs")
print("\n✅ Task 3 complete!")

"""
Iris Flower Species Classification
===================================
Internship Project - Task 1
CodeAlpha - Machine Learning with Scikit-learn

Classifies Iris flowers (setosa, versicolor, virginica)
based on sepal and petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────

print("=" * 60)
print("   IRIS FLOWER SPECIES CLASSIFICATION")
print("=" * 60)

df = pd.read_csv("Iris.csv")

# Drop the Id column — it adds no predictive value
df.drop(columns=["Id"], inplace=True)

print("\n📊 Dataset Overview")
print(f"   Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Classes: {df['Species'].unique().tolist()}")
print(f"   Balance:\n{df['Species'].value_counts().to_string()}\n")

# ─── 2. PREPARE FEATURES & LABELS ────────────────────────────────────────────

X = df.drop(columns=["Species"])
y = df["Species"]

# Encode string labels → integers (0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 80 / 20 train-test split, stratified so each class is equally represented
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✂️  Train samples : {len(X_train)}")
print(f"   Test  samples : {len(X_test)}\n")

# ─── 3. TRAIN MODEL ──────────────────────────────────────────────────────────

print("🌲 Training Random Forest Classifier …")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   ✅ Training complete!\n")

# ─── 4. EVALUATE ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"📈 Model Accuracy : {acc * 100:.2f}%\n")
print("📋 Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
    )
)

# ─── 5. VISUALISATIONS ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Iris Flower Classification — Model Analysis", fontsize=16, fontweight="bold")

# ── 5a. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[0, 0],
    linewidths=0.5,
)
axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
axes[0, 0].set_xlabel("Predicted Species")
axes[0, 0].set_ylabel("Actual Species")

# ── 5b. Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
importances.plot(kind="barh", ax=axes[0, 1], color=colors)
axes[0, 1].set_title("Feature Importances", fontweight="bold")
axes[0, 1].set_xlabel("Importance Score")
for bar, val in zip(axes[0, 1].patches, importances):
    axes[0, 1].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

# ── 5c. Petal scatter (best separating pair)
species_palette = {
    "Iris-setosa": "#4C72B0",
    "Iris-versicolor": "#DD8452",
    "Iris-virginica": "#55A868",
}
for species, color in species_palette.items():
    mask = df["Species"] == species
    axes[1, 0].scatter(
        df.loc[mask, "PetalLengthCm"],
        df.loc[mask, "PetalWidthCm"],
        label=species,
        color=color,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
    )
axes[1, 0].set_title("Petal Length vs Petal Width", fontweight="bold")
axes[1, 0].set_xlabel("Petal Length (cm)")
axes[1, 0].set_ylabel("Petal Width (cm)")
axes[1, 0].legend(fontsize=8)

# ── 5d. Sepal scatter
for species, color in species_palette.items():
    mask = df["Species"] == species
    axes[1, 1].scatter(
        df.loc[mask, "SepalLengthCm"],
        df.loc[mask, "SepalWidthCm"],
        label=species,
        color=color,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
    )
axes[1, 1].set_title("Sepal Length vs Sepal Width", fontweight="bold")
axes[1, 1].set_xlabel("Sepal Length (cm)")
axes[1, 1].set_ylabel("Sepal Width (cm)")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("iris_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("💾 Saved: iris_analysis.png")

# ─── 6. SAMPLE PREDICTION ────────────────────────────────────────────────────

print("\n🔮 Sample Prediction:")
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
)
pred = le.inverse_transform(model.predict(sample))[0]
probs = model.predict_proba(sample)[0]
print(f"   Input  : {sample.values[0].tolist()}")
print(f"   Prediction : {pred}")
print("   Confidence :")
for cls, prob in zip(le.classes_, probs):
    print(f"     {cls:<20} {prob * 100:.1f}%")

print("\n✅ All done! Check iris_analysis.png for the visualisations.")

# CodeAlpha Internship Tasks
Machine Learning & Data Analysis projects completed during the CodeAlpha internship.

---

## Task 1 — Iris Flower Classification

Classified Iris flower species (Setosa, Versicolor, Virginica) using sepal and petal measurements.

**Model:** Random Forest Classifier  
**Accuracy:** 90%  
**Libraries:** scikit-learn, pandas, matplotlib, seaborn

| Feature | Importance |
|---|---|
| Petal Length | Highest |
| Petal Width | High |
| Sepal Length | Medium |
| Sepal Width | Low |

**Files:** `iris_classification.py`, `Iris.csv`, `iris_analysis.png`

---

## Task 2 — Unemployment Analysis with Python

Analyzed unemployment trends in India using two datasets covering May 2019 – October 2020, with a focus on the Covid-19 impact.

**Key Findings:**
- Unemployment jumped from **9.5% → 17.8%** after the March 2020 lockdown (+87%)
- Peak unemployment hit **76.7%** in April 2020
- Urban areas were hit harder than rural areas
- Hardest hit states: Puducherry, Jharkhand, Haryana

**Libraries:** pandas, matplotlib, seaborn

**Files:** `unemployment_analysis.py`, `Unemployment_in_India.csv`, `Unemployment_Rate_upto_11_2020.csv`, `unemployment_analysis.png`, `unemployment_geo_analysis.png`

---

## Task 3 — Car Price Prediction with Machine Learning

Trained regression models to predict used car selling prices based on features like car age, present price, kilometers driven, fuel type, and transmission.

**Best Model:** Gradient Boosting Regressor  
**R² Score:** 0.9672 (96.7% variance explained)  
**Mean Absolute Error:** ₹0.53 Lakhs  

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.8468 | ₹1.22L |
| Random Forest | 0.9623 | ₹0.63L |
| Gradient Boosting ✅ | 0.9672 | ₹0.53L |

**Libraries:** scikit-learn, pandas, matplotlib, seaborn

**Files:** `car_price_prediction.py`, `car_data.csv`, `car_price_analysis.png`

---

## Setup & Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run any task
python iris_classification.py
python unemployment_analysis.py
python car_price_prediction.py
```

---

*Internship at [CodeAlpha](https://www.codealpha.tech/)*

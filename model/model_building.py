# model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ---------------- 1. LOAD DATA ----------------
df = pd.read_csv("train.csv")

# ---------------- 2. SELECT FEATURES ----------------
# Pick 6 features
features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt"]
target = "SalePrice"

X = df[features]
y = df[target]

# ---------------- 3. HANDLE MISSING VALUES ----------------
X.fillna(X.median(), inplace=True)  # Fill numeric missing values with median

# ---------------- 4. FEATURE SCALING ----------------
# For tree-based models like Random Forest, scaling is optional
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- 5. TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- 6. TRAIN MODEL ----------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- 7. EVALUATE MODEL ----------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# ---------------- 8. SAVE MODEL & SCALER ----------------
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("house_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")

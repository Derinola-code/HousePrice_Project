import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
try:
    # Look for train.csv in the root folder
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    # Backup: look for it in the current folder
    df = pd.read_csv('model/train.csv')

# 2. Feature Selection (Selecting 6 as per project requirements)
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
X = df[features]
y = df['SalePrice']

# 3. Preprocessing: Handle missing values
X = X.fillna(X.median())

# 4. Feature Scaling (Mandatory since you have scaler.pkl)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save Model and Scaler
if not os.path.exists('model'):
    os.makedirs('model')

with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Model trained. R2 Score: {r2_score(y_test, model.predict(X_test)):.4f}")
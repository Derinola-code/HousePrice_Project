import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction System")

st.write("### Property Specifications")

# Input fields for the 6 features
col1, col2 = st.columns(2)

with col1:
    qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    area = st.number_input("Gr. Living Area (sqft)", value=1500)
    bsmt = st.number_input("Total Basement (sqft)", value=1000)

with col2:
    garage = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    year = st.number_input("Year Built", value=2000)

if st.button("Calculate Predicted Price"):
    # 1. Prepare data
    input_data = np.array([[qual, area, bsmt, garage, bath, year]])
    
    # 2. Scale data (Matching training)
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)
    
    st.success(f"### Estimated Sale Price: ${prediction[0]:,.2f}")
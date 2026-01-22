import streamlit as st
import numpy as np
import pickle

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="🏡 House Price Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- 2. CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        color: #1a1a1a;
        font-family: 'Helvetica', sans-serif;
    }
    h1 {
        color: #ff6f61;
        text-align: center;
        font-size: 40px;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSlider > div > div > div > div {
        color: #ff6f61;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #ff6f61;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- 3. LOAD MODEL & SCALER ----------------
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("house_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- 4. SIDEBAR INPUT ----------------
st.sidebar.header("🏠 Input House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 300, 10000, 1500)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 5000, 800)
garage_cars = st.sidebar.number_input("Garage Capacity (cars)", 0, 5, 1)
full_bath = st.sidebar.number_input("Number of Full Bathrooms", 0, 5, 1)
year_built = st.sidebar.number_input("Year Built", 1800, 2026, 1990)

# ---------------- 5. MAIN HEADER ----------------
st.markdown("<h1>💰 House Price Prediction System</h1>", unsafe_allow_html=True)
st.markdown("### Enter house details in the sidebar to predict the price", unsafe_allow_html=True)
st.divider()

# ---------------- 6. PREDICTION ----------------
if st.button("Predict Price"):
    input_data = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    input_scaled = scaler.transform(input_data)
    
    price = model.predict(input_scaled)[0]
    
    st.markdown(
        f"""
        <div style='background-color:#ff6f61; padding:20px; border-radius:15px; text-align:center;'>
            <h2 style='color:white;'>🏷 Predicted House Price: ${price:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Made with 💙 using Streamlit & Scikit-learn")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load Keras model properly
model = load_model("model.h5", compile=False)

# Mapping dictionary (must match training encoding)
risk_mapping = {"low": 0, "medium": 0.5, "high": 1}

st.title("🏦 Bankruptcy Prevention System")
st.header("Enter Company Financial Data")

industrial_risk = st.selectbox("Industrial Risk", ["low", "medium", "high"])
management_risk = st.selectbox("Management Risk", ["low", "medium", "high"])
financial_flexibility = st.selectbox("Financial Flexibility", ["low", "medium", "high"])
credibility = st.selectbox("Credibility", ["low", "medium", "high"])
competitiveness = st.selectbox("Competitiveness", ["low", "medium", "high"])
operating_risk = st.selectbox("Operating Risk", ["low", "medium", "high"])

if st.button("Predict"):
    # Convert inputs to numeric
    input_data = np.array([[risk_mapping[industrial_risk],
                            risk_mapping[management_risk],
                            risk_mapping[financial_flexibility],
                            risk_mapping[credibility],
                            risk_mapping[competitiveness],
                            risk_mapping[operating_risk]]])

    # Predict probability and cast to float
    prediction = float(model.predict(input_data)[0][0])

    # Classification
    if prediction >= 0.5:
        st.error(f"⚠ Bankruptcy Risk Detected (Score: {prediction:.2f})")
    else:
        st.success(f"✅ Financially Safe (Score: {prediction:.2f})")

import streamlit as st
import numpy as np
import pandas as pd
import itertools
from tensorflow.keras.models import load_model

# Load Keras model properly
model = load_model("model.h5", compile=False)

# Mapping dictionary (must match training encoding)
risk_mapping = {"low": 0, "medium": 0.5, "high": 1}
categories = ["low", "medium", "high"]

# Features
features = [
    "Industrial Risk",
    "Management Risk",
    "Financial Flexibility",
    "Credibility",
    "Competitiveness",
    "Operating Risk"
]

# App Title
st.title("🏦 Bankruptcy Prevention System")

# Tabs for usability
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Explore All Combinations"])

# ---------------------- Single Prediction ----------------------
with tab1:
    st.header("Enter Company Financial Data")

    industrial_risk = st.selectbox("Industrial Risk", categories)
    management_risk = st.selectbox("Management Risk", categories)
    financial_flexibility = st.selectbox("Financial Flexibility", categories)
    credibility = st.selectbox("Credibility", categories)
    competitiveness = st.selectbox("Competitiveness", categories)
    operating_risk = st.selectbox("Operating Risk", categories)

    if st.button("Predict Bankruptcy Risk"):
        # Convert inputs to numeric
        input_data = np.array([[risk_mapping[industrial_risk],
                                risk_mapping[management_risk],
                                risk_mapping[financial_flexibility],
                                risk_mapping[credibility],
                                risk_mapping[competitiveness],
                                risk_mapping[operating_risk]]])

        # Predict probability and cast to float
        prediction = float(model.predict(input_data, verbose=0)[0][0])

        # Classification
        if prediction >= 0.5:
            st.error(f"⚠ Bankruptcy Risk Detected (Score: {prediction:.2f})")
        else:
            st.success(f"✅ Financially Safe (Score: {prediction:.2f})")

# ---------------------- Explore All Combinations ----------------------
with tab2:
    st.header("Explore All Possible Combinations")

    if st.button("Generate All Predictions"):
        # Generate all combinations (3^6 = 729 combos)
        combinations = list(itertools.product(categories, repeat=6))
        results = []

        for combo in combinations:
            # Convert to numeric input
            input_data = np.array([[risk_mapping[val] for val in combo]])
            
            # Predict probability
            prob = float(model.predict(input_data, verbose=0)[0][0])
            
            # Bankruptcy if prob >= 0.5
            label = "Bankruptcy" if prob >= 0.5 else "Non-Bankruptcy"
            
            results.append(list(combo) + [prob, label])

        # Create DataFrame
        df = pd.DataFrame(results, columns=features + ["Probability", "Prediction"])

        # Show DataFrame in Streamlit
        st.dataframe(df)

        # Option to download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "bankruptcy_combinations.csv", "text/csv")

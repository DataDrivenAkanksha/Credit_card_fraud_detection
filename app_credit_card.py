import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


#Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered",innclude_colab_link=True)

st.title("💳 Credit Card Fraud Detection App")
st.write("This app predicts whether a credit card transaction is **fraudulent or genuine** using 30 numerical features.")


# Load Model

model = joblib.load("random_forest.pkl")

# Input Method Selection

option = st.radio("Select Input Method", ["Upload CSV", "Manual Entry"])

# Standard Scaler for Time and Amount

scaler = joblib.load("scaler.pkl")

# CSV Upload

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your transaction CSV file (30 features, no Class column)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Check for correct columns
        expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in data.columns for col in expected_cols):
            st.error(f"❌ Uploaded file must contain the following columns:\n{expected_cols}")
        else:
            st.write("✅ Uploaded Data Preview:")
            st.dataframe(data.head())

            # Standard scale Time and Amount
            data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])

            # Predict
            if st.button("Predict"):
                predictions = model.predict(data)
                data['Predicted Class'] = predictions
                st.success("🎯 Prediction Complete!")
                st.dataframe(data)

# Manual Entry

else:
    st.subheader("Enter Transaction Details")

    time = st.number_input("Time", min_value=0.0, value=172792.0)
    amount = st.number_input("Amount", min_value=0.0, value=50000.0)

    v_features = {}
    for i in range(1, 29):
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

    # Combine inputs
    input_data = pd.DataFrame([{"Time": time, **v_features, "Amount": amount}])

    # Apply standard scaling to Time and Amount
    input_data[['Time', 'Amount']] = scaler.transform(input_data[['Time', 'Amount']])

    st.write("Input Summary:")
    st.dataframe(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

st.markdown("**Note:** Model uses 30 input columns — `Time`, `V1–V28`, and `Amount`. The app predicts the `Class` output (0 = Legitimate, 1 = Fraud).")
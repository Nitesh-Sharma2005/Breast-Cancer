import streamlit as st
import numpy as np
import joblib

# Load trained model & scaler
model = joblib.load("Breast_Cancer.joblib")
scaler = joblib.load("scaler_Breast_Cancer.joblib")

st.set_page_config(page_title="Breast Cancer Diagnosis", layout="centered")
st.title("🩺 Breast Cancer Diagnosis Prediction")

st.write("Fill **all fields** to predict whether the tumor is Benign (0) or Malignant (1).")

# ---------- Input Section ----------
def get_user_input():
    return np.array([
        st.number_input("Radius Mean", min_value=0.0),
        st.number_input("Texture Mean", min_value=0.0),
        st.number_input("Perimeter Mean", min_value=0.0),
        st.number_input("Area Mean", min_value=0.0),
        st.number_input("Smoothness Mean", min_value=0.0),
        st.number_input("Compactness Mean", min_value=0.0),
        st.number_input("Concavity Mean", min_value=0.0),
        st.number_input("Concave Points Mean", min_value=0.0),
        st.number_input("Symmetry Mean", min_value=0.0),

        st.number_input("Radius SE", min_value=0.0),
        st.number_input("Perimeter SE", min_value=0.0),
        st.number_input("Area SE", min_value=0.0),
        st.number_input("Compactness SE", min_value=0.0),
        st.number_input("Concavity SE", min_value=0.0),
        st.number_input("Concave Points SE", min_value=0.0),

        st.number_input("Radius Worst", min_value=0.0),
        st.number_input("Texture Worst", min_value=0.0),
        st.number_input("Perimeter Worst", min_value=0.0),
        st.number_input("Area Worst", min_value=0.0),
        st.number_input("Smoothness Worst", min_value=0.0),
        st.number_input("Compactness Worst", min_value=0.0),
        st.number_input("Concavity Worst", min_value=0.0),
        st.number_input("Concave Points Worst", min_value=0.0),
        st.number_input("Symmetry Worst", min_value=0.0),
        st.number_input("Fractal Dimension Worst", min_value=0.0)
    ]).reshape(1, -1)

input_data = get_user_input()

# ---------- Prediction ----------
if st.button("🔍 Predict Diagnosis"):

    if np.any(input_data == 0):
        st.warning("⚠️ Please enter all values before prediction.")
    else:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)

        if prediction[0] == 1:
            st.error("⚠️ Malignant Tumor Detected")
        else:
            st.success("✅ Benign Tumor Detected")

        st.subheader("Prediction Probability")
        st.write(f"Benign (0): {probability[0][0]:.4f}")
        st.write(f"Malignant (1): {probability[0][1]:.4f}")

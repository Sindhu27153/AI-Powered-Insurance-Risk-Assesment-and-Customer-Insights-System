import streamlit as st
import numpy as np
import pickle

def run_risk_score_interface():
    st.title("💡 Risk Score Prediction")
    st.markdown("✍️ Enter Customer Details to Predict Risk Score")

    # Input fields
    age = st.slider("Customer Age", 18, 100, 30)
    income = st.number_input("Annual Income (₹)", min_value=10000, value=500000)
    vehicle_age = st.slider("Vehicle or Property Age (in years)", 0, 30, 5)
    claim_history = st.slider("Claim History (number of past claims)", 0, 20, 2)
    premium = st.number_input("Premium Amount (₹)", min_value=1000, value=15000)
    
    # Engineered Features
    claim_income_ratio = claim_history / (income + 1)
    premium_income_ratio = premium / (income + 1)
    high_claim = int(claim_history > 5)
    claim_premium_diff = claim_history - premium

    # Gender and Policy Type - One-hot encoding
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    policy_type = st.selectbox("Policy Type", ["Health", "Life", "Property"])
    age_group = st.selectbox("Age Group", ["Adult", "Mid-Age", "Senior"])

    # One-hot encoded vectors
    gender_male = int(gender == "Male")
    gender_other = int(gender == "Other")
    policy_health = int(policy_type == "Health")
    policy_life = int(policy_type == "Life")
    policy_property = int(policy_type == "Property")
    age_adult = int(age_group == "Adult")
    age_mid = int(age_group == "Mid-Age")
    age_senior = int(age_group == "Senior")

    # Final input feature vector
    input_features = np.array([[
        age, income, vehicle_age, claim_history, premium,
        claim_income_ratio, premium_income_ratio, high_claim, claim_premium_diff,
        gender_male, gender_other,
        policy_health, policy_life, policy_property,
        age_adult, age_mid, age_senior
    ]])

    # Load model and scaler
    model_path = "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/model_risk_score.pkl"
    scaler_path = "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/scaler_risk.pkl"

    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError:
        st.error("❌ Model or Scaler file not found. Please check the file paths.")
        return

    if st.button("🔍 Predict Risk Score"):
        try:
            input_scaled = scaler.transform(input_features)
            prediction = model.predict(input_scaled)[0]

            # Map prediction to label
            label_mapping = {0: "Low", 1: "Medium", 2: "High"}
            predicted_label = label_mapping.get(prediction, "Unknown")

            st.success(f"🎯 Predicted Risk Score: {predicted_label} ({prediction})")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")


if __name__ == "__main__":
    run_risk_score_interface()

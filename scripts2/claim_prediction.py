import streamlit as st
import pickle
import numpy as np

def run_claim_prediction_interface():
    st.title("💰 Insurance Claim Amount Prediction")
    st.markdown("Enter customer information to predict the **expected insurance claim amount**.")

    # Load model and scaler
    with open("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/model_claim_amount_prediction.pkl", "rb") as f_model:
        model = pickle.load(f_model)

    with open("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/scaler_claim.pkl", "rb") as f_scaler:
        scaler = pickle.load(f_scaler)

    # ----------------- Input Fields -----------------
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    annual_income = st.number_input("Annual Income (₹)", min_value=0, value=500000)
    vehicle_or_property_age = st.number_input("Vehicle/Property Age (years)", min_value=0, value=5)
    claim_history = st.number_input("Number of Past Claims", min_value=0, value=1)

    fraudulent_claim = st.selectbox("Is Fraudulent Claim Suspected?", ['No', 'Yes'])
    fraudulent_claim_val = 1 if fraudulent_claim == 'Yes' else 0

    premium_amount = st.number_input("Premium Amount Paid (₹)", min_value=0, value=25000)
    premium_income_ratio = premium_amount / annual_income if annual_income > 0 else 0

    # Gender
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    gender_male = 1 if gender == 'Male' else 0
    gender_other = 1 if gender == 'Other' else 0

    # Policy Type
    policy_type = st.selectbox("Policy Type", ['Life', 'Health', 'Property'])
    policy_type_life = 1 if policy_type == 'Life' else 0
    policy_type_health = 1 if policy_type == 'Health' else 0
    policy_type_property = 1 if policy_type == 'Property' else 0

    # Age Group
    age_group = st.selectbox("Age Group", ['Adult', 'Mid-Age', 'Senior'])
    age_group_adult = 1 if age_group == 'Adult' else 0
    age_group_mid = 1 if age_group == 'Mid-Age' else 0
    age_group_senior = 1 if age_group == 'Senior' else 0

    # Risk Score Label
    risk_score_label = st.selectbox("Predicted Risk Score Label", [0, 1, 2])  # 0 = Low, 1 = Medium, 2 = High Risk

    # ----------------- Prediction -----------------
    if st.button("🔍 Predict Claim Amount"):
        input_data = np.array([[
            customer_age,
            annual_income,
            vehicle_or_property_age,
            claim_history,
            fraudulent_claim_val,
            premium_amount,
            premium_income_ratio,
            gender_male,
            gender_other,
            policy_type_health,
            policy_type_life,
            policy_type_property,
            age_group_adult,
            age_group_mid,
            age_group_senior,
            risk_score_label
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(f"💸 Predicted Claim Amount: ₹{prediction:,.2f}")
        st.info("This is an estimated claim amount based on the provided inputs.")


if __name__ == "__main__":
    run_claim_prediction_interface()




    

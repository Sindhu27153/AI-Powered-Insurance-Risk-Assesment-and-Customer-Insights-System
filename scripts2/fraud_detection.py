import streamlit as st
import pandas as pd
import pickle

# Title of the app


# Load the model and scaler
with open("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/model_Fraudulent_claim1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler (you need to save the scaler after training)
with open("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/scaler_fraudulent_claim.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to clean column names (same as in training)
def clean_column_names(df):
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_-]+', '_', regex=True)
        .str.strip('_')
    )
    return df

# Input form for user to manually provide the data
def user_input():
    st.subheader("Enter the details for the claim:")
    
    # Customer Age
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

    # Annual Income
    annual_income = st.number_input("Annual Income (in $)", min_value=0, value=50000)

    # Vehicle or Property Age
    vehicle_age = st.number_input("Vehicle or Property Age (in years)", min_value=0, value=5)

    # Claim History (number of previous claims)
    claim_history = st.number_input("Claim History (Number of Previous Claims)", min_value=0, value=0)

    # Premium Amount
    premium_amount = st.number_input("Premium Amount (in $)", min_value=0, value=500)

    # Claim Income Ratio
    claim_income_ratio = st.number_input("Claim Income Ratio", min_value=0.0, value=0.1, format="%.2f")

    # Premium Income Ratio
    premium_income_ratio = st.number_input("Premium Income Ratio", min_value=0.0, value=0.05, format="%.2f")

    # High Claim (1 for high, 0 for low)
    high_claim = st.selectbox("High Claim (Yes=1, No=0)", options=[0, 1])

    # Claim Premium Difference
    claim_premium_diff = st.number_input("Claim Premium Difference", min_value=0.0, value=50.0, format="%.2f")

    # Gender
    gender = st.radio("Gender", options=["Male", "Female", "Other"])

    # Policy Type
    policy_type = st.selectbox("Policy Type", options=["Health", "Life", "Property"])

    # Age Group
    age_group = st.selectbox("Age Group", options=["Adult", "Mid-Age", "Senior"])

    # Prepare data for prediction
    data = {
        'Customer_Age': customer_age,
        'Annual_Income': annual_income,
        'Vehicle_or_Property_Age': vehicle_age,
        'Claim_History': claim_history,
        'Premium_Amount': premium_amount,
        'Claim_Income_Ratio': claim_income_ratio,
        'Premium_Income_Ratio': premium_income_ratio,
        'High_Claim': high_claim,
        'Claim_Premium_Diff': claim_premium_diff,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Other': 1 if gender == 'Other' else 0,
        'Policy_Type_Health': 1 if policy_type == 'Health' else 0,
        'Policy_Type_Life': 1 if policy_type == 'Life' else 0,
        'Policy_Type_Property': 1 if policy_type == 'Property' else 0,
        'Age_Group_Adult': 1 if age_group == 'Adult' else 0,
        'Age_Group_Mid-Age': 1 if age_group == 'Mid-Age' else 0,
        'Age_Group_Senior': 1 if age_group == 'Senior' else 0
    }

    input_data = pd.DataFrame(data, index=[0])
    return input_data

# Function to predict fraudulent claim
def predict_claim(input_data):
    # Clean column names to match the model's expectations
    input_data = clean_column_names(input_data)

    # Ensure the input data has the same columns as the model's expected features
    expected_columns = ['Customer_Age', 'Annual_Income', 'Vehicle_or_Property_Age', 'Claim_History', 'Premium_Amount', 
                        'Claim_Income_Ratio', 'Premium_Income_Ratio', 'High_Claim', 'Claim_Premium_Diff', 
                        'Gender_Male', 'Gender_Other', 'Policy_Type_Health', 'Policy_Type_Life', 'Policy_Type_Property', 
                        'Age_Group_Adult', 'Age_Group_Mid-Age', 'Age_Group_Senior']
    
    # Reorder columns to match expected feature order
    input_data = input_data[expected_columns]

    if input_data is not None:
        # Apply scaling to the features (same scaler used during training)
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)

        # Display the prediction
        if prediction == 1:
            st.success("⚠️ This claim is likely fraudulent!")
        else:
            st.success("✅ This claim is likely legitimate.")         
    else:
        st.error("The input data does not match the model's expected features. Please check the input fields.")

# Streamlit Interface
def run_fraud_detection_interface():
    st.title("🚨 Fraud Detection System")

    # Show the user input form
    input_data = user_input()
    
    # Button to make the prediction
    if st.button("Predict Fraudulent Claim"):
        predict_claim(input_data)

# Run the Streamlit app
if __name__ == "__main__":
    run_fraud_detection_interface()

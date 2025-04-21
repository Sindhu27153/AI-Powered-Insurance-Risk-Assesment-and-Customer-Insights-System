import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Drop Policy_ID if it exists
    if 'Policy_ID' in df.columns:
        df.drop("Policy_ID", axis=1, inplace=True)

    # Handle missing values based on skewness
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                if abs(df[col].skew()) > 1:
                    df[col].fillna(df[col].median())
                else:
                    df[col].fillna(df[col].mean())
            else:
                df[col].fillna(df[col].mode()[0])

    # Outlier removal using IQR for numerical columns
    num_cols = ['Customer_Age', 'Annual_Income', 'Vehicle_or_Property_Age', 
                'Claim_History', 'Premium_Amount', 'Claim_Amount']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # One-hot encoding for categorical variables
    categorical_cols = [col for col in ['Gender', 'Policy_Type'] if col in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    

    # Feature engineering
    df['Claim_Income_Ratio'] = df['Claim_Amount'] / df['Annual_Income']
    df['Premium_Income_Ratio'] = df['Premium_Amount'] / df['Annual_Income']
    df['Claim_Premium_Diff'] = df['Claim_Amount'] - df['Premium_Amount']
    df['High_Claim'] = df['Claim_Amount'] > df['Premium_Amount']

    # Age group classification
    df['Age_Group_Adult'] = df['Customer_Age'].apply(lambda x: 1 if x < 30 else 0)
    df['Age_Group_Mid-Age'] = df['Customer_Age'].apply(lambda x: 1 if 30 <= x < 60 else 0)
    df['Age_Group_Senior'] = df['Customer_Age'].apply(lambda x: 1 if x >= 60 else 0)

    # Encode Risk_Score_Label if present
    if 'Risk_Score' in df.columns:
        mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Risk_Score_Label'] = df['Risk_Score'].map(mapping)

    # Convert High_Claim to integer
    df['High_Claim'] = df['High_Claim'].astype(int)

    #drop original columns
    cols_to_drop = [col for col in ['Risk_Score', 'Customer_Age'] if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    return df


if __name__ == "__main__":
    df_cleaned = load_and_clean_data("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Dataset/Insurance_data.csv")
    print("✅ Preprocessing complete. Shape:", df_cleaned.shape)
    
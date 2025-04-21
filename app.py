import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts2.chatbot import chatbot_interface
from scripts2.risk_score import run_risk_score_interface
from scripts2.claim_prediction import run_claim_prediction_interface
from scripts2.sentiment_analysis import run_sentiment_analysis_interface
from scripts2.fraud_detection import run_fraud_detection_interface
from scripts2.translate_and_summarize import run_translator_interface  


# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.radio("Go to", (
    "🏠 Home",
    "📊 Risk Score Prediction",
    "💰 Claim Amount Prediction",
    "🚨 Fraud Detection",
    "💬 Sentiment Analysis",
    "🌐 Policy Translation & Summarization",
    "🤖 FAQ Chatbot",
    "📈 Exploratory Data Analysis (EDA)"  # Added EDA option
))

# Function to load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Dataset/Insurance_data.csv")  # Replace with your dataset path
    return df

# Function to display EDA

@st.cache_data
def display_eda(df):
    st.title("🔍 Exploratory Data Analysis (EDA)")

    # Show basic stats
    st.subheader("📊 Basic Statistics")
    st.write(df.describe())


    # Interactive plots
    st.subheader("📈 Claim Amount Distribution")
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(df['Claim_Amount'], kde=True, color="blue")
    st.pyplot(fig)

    #Fradulent Claims Distribution
    st.subheader("🚩 Distribution of Fraudulent Claims")
    fig, ax = plt.subplots()
    df["Fraudulent_Claim"].value_counts().plot(kind='bar', colormap='viridis', ax=ax)
    ax.set_ylabel("Counts")
    ax.set_title("Distribution of Fradulent Claims")
    st.pyplot(fig)
    

    #Gender Distribution
    st.subheader("👥 Distribution of Policy Holders by Gender")
    fig, ax = plt.subplots()
    df["Gender"].value_counts().plot(kind='bar', colormap='coolwarm', ax=ax)
    ax.set_ylabel("Counts")
    ax.set_title("Distribution Of Policy Holders")
    st.pyplot(fig)

    # Policy Type Distribution
    st.subheader("📂 Distribution of Policy Types")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Policy_Type", palette='coolwarm', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Claim Amount by Policy Type
    st.subheader("💰 Claim Amount by Policy Type")
    fig,ax=plt.subplots()
    sns.barplot(x='Policy_Type', y='Claim_Amount', data=df, alpha=0.7,color='red')
    plt.xticks(rotation=45)
    plt.xlabel("Policy Type")
    st.pyplot(fig)

    # Annual Income vs Claim Amount
    st.subheader("💵 Annual Income vs Claim Amount")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Annual_Income', y='Claim_Amount', data=df, ax=ax, color='purple')
    ax.set_title("Annual Income vs Claim Amount")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Claim Amount")
    st.pyplot(fig)


    #Show correlation heatmap
    st.subheader("🧩 Correlation Heatmap")
    df.drop(columns=['Policy_ID'], inplace=True, errors='ignore')  # Drop non-numeric columns for correlation
    df=df.select_dtypes(exclude=["object"])  # Select only numeric columns for correlation
    correlation_matrix = df.corr()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

    st.write("This heatmap shows the correlation between different features in the dataset. A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation. Values around 0 indicate no correlation.")

# Home Page
if app_mode == "🏠 Home":
    st.title("🧠 AI-Powered Insurance Risk Assessment & Customer Insights")
    st.markdown("""
    Welcome to the **AI-Powered Insurance Risk Assessment and Customer Insights Dashboard**!  
    This intelligent platform helps insurers and analysts make smarter, faster, and more accurate decisions using the power of machine learning and natural language processing.

    ---
    ## 🔍 Key Features

    - ✅ **Risk Score Prediction**  
      Analyze customer and policy data to assess risk levels with advanced classification models.

    - 💰 **Claim Amount Prediction**  
      Estimate potential claim payouts using predictive regression models for better financial planning.

    - 🚨 **Fraud Detection**  
      Identify potentially fraudulent claims in real-time using powerful anomaly detection algorithms.

    - 💬 **Sentiment Analysis**  
      Analyze customer feedback and complaints to uncover hidden insights and improve customer satisfaction.

    - 📄 **Policy Translation & Summarization**  
      Translate insurance policies into multiple languages and summarize them for easy understanding.

    - 🤖 **AI Chatbot for FAQs**  
      Instantly answer customer queries with our multilingual insurance chatbot.

    ---
    ## 🎯 Who is this for?

    - **Insurance Companies**
    - **Underwriters and Actuaries**
    - **Policy Analysts**
    - **Customer Service Teams**
    - **Data Scientists**

    ---
    ## 📊 Powered By

    - Python 🐍 | Streamlit 📈 | Scikit-learn | LightGBM | XGBoost  
    - NLP with spaCy, TextBlob, and NLTK  
    - Multilingual support with Google Translate API  
    - Machine Learning + Data Visualization = 🔥

    ---
    Let data drive your decisions. Make insurance smarter.  
    **Start exploring from the sidebar →**
    """)

# Load the selected module
elif app_mode == "🤖 FAQ Chatbot":
    chatbot_interface()

elif app_mode == "🌐 Policy Translation & Summarization":
    st.header("Policy Translator and Summarizer")
    st.write("Upload your insurance policy PDF and select the languages for translation and summarization.")

    # File uploader for the PDF
    pdf_file = st.file_uploader(
        "Upload Insurance Policy PDF", 
        type=["pdf"], 
        key="policy_uploader"
    )

    # Language selector for translation
    languages = st.multiselect(
        "Select languages to translate the policy into",
        options=["Hindi", "Tamil", "Telugu", "Gujarati", "Kannada", "Bengali", "Malayalam", "Urdu"]
    )

    # Button to trigger translation and summarization
    if pdf_file and languages:
        if st.button("Translate and Summarize"):
            # Save the uploaded PDF to a temporary path
            temp_dir = "temp"
            pdf_path = os.path.join(temp_dir, pdf_file.name)
            os.makedirs(temp_dir, exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Provide feedback to the user
            st.info("Translating and summarizing your document... please wait.")

            try:
                # Call the function to handle both translation and summarization
                run_translator_interface(uploaded_file=pdf_file, selected_languages=languages)


                st.success("Translation and summarization completed!")

                # Display download buttons for translated PDFs
                translated_dir = "Translated_Policies"
                for lang in languages:
                    translated_pdf_path = os.path.join(translated_dir, f"Translated_Insurance_Policy_{lang}.pdf")
                    
                    # Check if the PDF exists
                    if os.path.exists(translated_pdf_path):
                        with open(translated_pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label=f"Download {lang} PDF",
                                data=pdf_file.read(),
                                file_name=f"Translated_Insurance_Policy_{lang}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.warning(f"❌ {lang} translation failed or PDF not found.")
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")

    elif not pdf_file:
        st.warning("Please upload an insurance policy PDF.")

    elif not languages:
        st.warning("Please select at least one language for translation.")

elif app_mode == "📈 Exploratory Data Analysis (EDA)":
    # Load and display the dataset for EDA
    df = load_data()
    display_eda(df)

elif app_mode == "📊 Risk Score Prediction":
    run_risk_score_interface()

elif app_mode == "💰 Claim Amount Prediction":
    run_claim_prediction_interface()

elif app_mode == "💬 Sentiment Analysis":
    run_sentiment_analysis_interface()

elif app_mode == "🚨 Fraud Detection":
    run_fraud_detection_interface()

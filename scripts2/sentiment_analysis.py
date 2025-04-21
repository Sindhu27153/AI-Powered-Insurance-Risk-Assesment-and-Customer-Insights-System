import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import label_binarize
import pickle
import streamlit as st


# Step 1: Text Preprocessing Functions
def clean_text(text):
    text = re.sub(r'#|\$\$|##|\n', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load model and vectorizer with error handling
try:
    with open('D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/model_sentimental_analysis.pkl', 'rb') as model_file:
        model_nb = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()  # Stops further execution if the model is not found.

try:
    with open('D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Models/tfidfvectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("TF-IDF Vectorizer file not found. Please check the file path.")
    st.stop()  # Stops further execution if the vectorizer is not found.

# Step 2: Streamlit Interface
def run_sentiment_analysis_interface():
    st.title("Sentiment Analysis for Insurance Feedback")
    user_input = st.text_area("Enter your feedback", "")

    if st.button("Analyze Sentiment"):
        if user_input:
            cleaned_input = preprocess_text(user_input)
            input_vector = vectorizer.transform([cleaned_input]).toarray()

            prediction = model_nb.predict(input_vector).item()  # Extract scalar value

            if prediction == 1:
                st.write("Sentiment: **Positive**")
            else:
                st.write("Sentiment: **Negative**")
        else:
            st.write("Please enter some feedback.")

# Run the Streamlit app
if __name__ == "__main__":
    run_sentiment_analysis_interface()


# 🧠 AI-Powered Insurance Risk Assessment & Customer Insights System

A smart, interactive dashboard built using Streamlit that utilizes machine learning, NLP, and data visualization to transform raw insurance data into actionable business insights.

---

## 📌 Executive Summary

This AI-powered system helps insurance providers automate and improve their decision-making pipeline. From identifying high-risk customers and predicting claims, to detecting fraud, understanding customer sentiment, and simplifying policy documents via translation and summarization—the platform offers an end-to-end solution powered by machine learning and natural language processing.



## 📊 Exploratory Data Analysis (EDA)

### Key Visual Insights:
- **Claim Amount Distribution**: Skewed distribution, with few high-value claims.
- **Fraudulent Claims**: Clear imbalance between fraudulent and non-fraudulent cases.
- **Policy Type Trends**: "Comprehensive" policies are the most common.
- **Gender Split**: Fairly balanced but with slight male dominance.
- **Income vs. Claim**: No strong linear correlation found.
- **Correlation Heatmap**: claim_amount and Premium_Amount slightly correlated with claims.



## 🧪 Model Training & Evaluation

### 1. Risk Score Prediction
- **Techniques Used**: Logistic Regression, Random Forest, XGBoost
- **Best Score**: 89.4% Accuracy with RandomForest
- **Skills Applied**: Classification modeling, Cross-validation, Feature engineering

### 2. Claim Amount Prediction
- **Techniques Used**: Linear Regression, Random Forest Regressor ,XGBoost, LightGBM
- **Best Score**: RMSE ≈ 150 (Random Forest)
- **Skills Applied**: Regression modeling, Data scaling, Model evaluation

### 3. Fraud Detection
- **Techniques Used**: Random Forest, Logistic Regression, XGBoost, LightGBM
- **Best Score**: 93.5% Accuracy (LightGBM)
- **Skills Applied**: Ensemble methods, SMOTE for class imbalance, Confusion matrix

### 4. Sentiment Analysis
- **Techniques Used**: TextBlob, NLTK
- **Accuracy**: 90% (naive Bayes)
- **Skills Applied**: Text pre-processing, polarity detection, feedback analysis

### 5. Translation & Summarization
- **Techniques Used**: Google Translate API, rule-based summarization
- **Output**: Multilingual translated policy PDFs
- **Skills Applied**: PDF processing, multilingual NLP, file handling

### 6. FAQ Chatbot
- **Techniques Used**: Rule-based + language detection
- **Features**: Supports multiple Indian languages
- **Skills Applied**: Intent classification, multilingual support, basic chatbot logic


## ⚠️ Challenges & Solutions

| Module             | Challenge                                              | Solution                                                             |
|-------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| Risk Scoring       | Class overlap, imbalance                              | Stratified cross-validation, class weighting                        |
| Claim Prediction   | Outliers, skewed data                                 | Log transformation, Random Forest for non-linearity                 |
| Fraud Detection    | Minority class hard to detect                         | Used SMOTE + ensemble methods (LightGBM & XGBoost)                  |
| Sentiment Analysis | Handling sarcasm, multilingual reviews                | Rule-based polarity + text cleaning, fallback language detection    |
| Translation        | Formatting issues in translated PDFs                  | Custom rendering with PyMuPDF + ReportLab                           |
| Chatbot            | Rule-based limits and edge cases                      | Intent mapping and multilingual fallback logic                      |


## 🔍 Customer Segmentation

- **Clustering Technique**: DBscan Clustering
- **Input Features**: Age, Annual Income, Policy Type, Claim Amount
- **Purpose**: Enable personalized marketing and premium pricing
- **Output**: Customers grouped into clusters with similar risk/behavior

---

## 💡 Future Enhancements

- Integrate Deep Learning models for sentiment and fraud detection
- Enable speech-to-text interaction with the chatbot
- Real-time analytics with dashboard refresh
- User login and report storage
- Cloud integration (AWS/GCP)

---

## 🗂️ Project Structure

```bash
Insurance_AI_Project  
├── 📂 data                # Raw & processed datasets  
├── 📂 notebooks           # Jupyter notebooks for EDA, ML, NLP models  
├── 📂 models              # Trained model files (Pickle, ONNX, TensorFlow, PyTorch)  
├── 📂 scripts             # Python scripts for data processing & model training  
├── 📂 deployment          # API files (Flask/FastAPI), Docker, Streamlit UI  
├── 📂 reports             # Project documentation & reports  
├── 📜 README.md           # Project overview & setup instructions  
├── 📜 requirements.txt    # Dependencies & libraries used  
└── 📜 app.py              # Main entry point for deployment  
```
---

## 🔧 Skills Demonstrated

- Machine Learning (Classification, Regression, Clustering)
- Natural Language Processing (Sentiment, Translation, Chatbot)
- Data Visualization (EDA)
- Model Evaluation & Hyperparameter Tuning
- Streamlit App Development
- PDF Generation & File Handling

---

## 🧾 Appendices (Optional)

### A. Data Overview

| Feature           | Type     | Description                                  |
|-------------------|----------|----------------------------------------------|
| Age               | Numeric  | Age of the policyholder                      |
| Gender            | Categorical | Male / Female                            |
| Annual Income     | Numeric  | Yearly income                                |
| Policy_Type       | Categorical | Type of policy held                        |
| Claim_Amount      | Numeric  | Claimed amount in currency                   |
| Fraudulent_Claim  | Binary   | Whether the claim is fraudulent              |

### B. Model Code Example

````python
# Example: Logistic Regression for Risk Score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

````


## 📚 References
Scikit-learn

Analytics Vidhya

Google Translate API

Streamlit Documentation

TextBlob

## 🙋 Contact
Developed by [Your Name]
📧 Email: sindhuja.ene@gmail.com
🔗 GitHub: github.com/27153


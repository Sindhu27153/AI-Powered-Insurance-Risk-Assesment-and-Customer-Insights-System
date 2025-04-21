import json
from fuzzywuzzy import process
from googletrans import Translator
import streamlit as st

# Initialize Google Translator
translator = Translator()

# Path to your FAQ file
faq_file = "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/Dataset/insurance_faq_100.json"

# Load FAQ data
with open(faq_file, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Extract all questions
questions = [faq["question"] for faq in faq_data]

# Language mapping
language_map = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Marathi": "mr"
}

# Function to find best answer using fuzzy matching
def get_answer(user_input_en):
    best_match, score = process.extractOne(user_input_en, questions)
    if score >= 60:
        for faq in faq_data:
            if faq["question"] == best_match:
                return faq["answer"]
    return "Sorry, I couldn't find an answer to that. Please try rephrasing your question."

# Streamlit interface
def chatbot_interface():
    st.title("🌐 Multilingual Insurance FAQ Chatbot")
    st.markdown("Ask any insurance-related question in your preferred language.")

    # Language selection
    selected_language = st.selectbox("Select Language", list(language_map.keys()))
    lang_code = language_map[selected_language]

    # Initialize or clean session state conversation
    if "conversation" not in st.session_state or not isinstance(st.session_state.conversation, list):
        st.session_state.conversation = []

    # Clean up old string-based entries from earlier runs
    if st.session_state.conversation and isinstance(st.session_state.conversation[0], str):
        st.session_state.conversation = []

    # Input area
    user_input = st.text_input("Type your question here:")

    # Ask button
    if st.button("Ask"):
        if user_input.strip() != "":
            try:
                # Translate user input to English
                translated_input = translator.translate(user_input, src=lang_code, dest='en').text

                # Get answer in English
                answer_en = get_answer(translated_input)

                # Translate answer back to selected language
                translated_answer = translator.translate(answer_en, src='en', dest=lang_code).text

                # Append to conversation history
                st.session_state.conversation.append(("You", user_input))
                st.session_state.conversation.append(("Chatbot", translated_answer))

            except Exception as e:
                st.error(f"Translation Error: {e}")
        else:
            st.warning("Please enter a question.")

    # Optional: Clear chat history
    if st.button("Clear Chat"):
        st.session_state.conversation = []

    # Display conversation history
    st.markdown("### 💬 Conversation History")
    for sender, message in st.session_state.conversation:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

# Run the app
if __name__ == "__main__":
    chatbot_interface()

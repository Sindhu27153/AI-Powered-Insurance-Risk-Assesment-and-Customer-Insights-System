import os
import fitz  # PyMuPDF
from fpdf import FPDF
from googletrans import Translator
import textwrap
import time
import streamlit as st

# Constants
FONT_PATH = "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSans-Regular.ttf"
SUPPORTED_LANGUAGES = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Marathi": "mr",
    "Malayalam": "ml",
    "Urdu": "ur",
    "Odia": "or"
}
TRANSLATED_DIR = "Translated_Policies"

# Translator utils
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc]).strip()

def safe_translate(translator, text, lang_code, retries=3, delay=2):
    """
    Attempts to translate the given text to a target language, retrying on failure.
    """
    for attempt in range(retries):
        try:
            return translator.translate(text, dest=lang_code).text
        except Exception as e:
            time.sleep(delay)  # Retry after a delay
    return None

class UnicodePDF(FPDF):
    def __init__(self, font_path, font_family, title=""):
        super().__init__()
        self.font_family = font_family
        self.title = title
        self.add_font("NotoSans", '', font_path, uni=True)
        self.add_page()
        self.set_font(font_family, '', 12)
        self.set_left_margin(10)
        self.set_right_margin(10)

    def header(self):
        self.set_font(self.font_family, '', 16)
        self.cell(0, 10, self.title, ln=True, align='C')
        self.ln(10)

    def add_multiline_text(self, text):
        # Break the text into lines, manually adjusting for long text
        text = text.replace("\n", " \n")  # Ensure that line breaks are respected
        wrapped_text = textwrap.fill(text, width=90)  # Adjust width for wrapping
        for paragraph in wrapped_text.split("\n"):
            self.multi_cell(0, 8, paragraph)
            self.ln(2)
    
    def save(self, output_path):
        self.output(output_path)

def create_translated_pdf(text, language, output_path):
    """
    Creates and saves a PDF with the translated text.
    """
    pdf = UnicodePDF(FONT_PATH, "NotoSans", f"Translated Insurance Policy - {language}")
    pdf.add_multiline_text(text)
    pdf.save(output_path)

def translate_text_for_language(text, lang_code):
    """
    Translates the extracted text into the selected language.
    """
    translator = Translator()
    translated = safe_translate(translator, text, lang_code)
    return translated

def save_translated_pdf(translated_text, lang, output_path):
    """
    Saves the translated text to a new PDF.
    """
    create_translated_pdf(translated_text, lang, output_path)
    print(f"✅ {lang} PDF saved: {output_path}")

def process_translation(pdf_path, selected_languages):
    """
    Main logic for processing translation of the given PDF into selected languages.
    """
    text = extract_text_from_pdf(pdf_path)
    
    os.makedirs(TRANSLATED_DIR, exist_ok=True)
    
    for lang in selected_languages:
        lang_code = SUPPORTED_LANGUAGES[lang]
        translated_text = translate_text_for_language(text, lang_code)
        
        if translated_text:
            output_path = os.path.join(TRANSLATED_DIR, f"Translated_Insurance_Policy_{lang}.pdf")
            save_translated_pdf(translated_text, lang, output_path)
        else:
            print(f"❌ Failed to translate to {lang}")

# The function you will call in your Streamlit app or other code
def run_translator_interface(pdf_path, selected_languages):
    """
    Interface function that translates PDF into selected languages and creates downloadable PDFs.
    """
    translator = Translator()
    text = extract_text_from_pdf(pdf_path)
    
    os.makedirs(TRANSLATED_DIR, exist_ok=True)
    
    # Add feedback to user interface (Streamlit)
    for lang in selected_languages:
        lang_code = SUPPORTED_LANGUAGES[lang]
        
        # Translate text
        translated_text = safe_translate(translator, text, lang_code)
        
        if translated_text:
            try:
                # Attempt to create the PDF for each language
                output_path = os.path.join(TRANSLATED_DIR, f"Translated_Insurance_Policy_{lang}.pdf")
                create_translated_pdf(translated_text, lang, output_path)
                print(f"✅ {lang} PDF saved: {output_path}")
                
                # Here you can include a Streamlit button or link for users to download the file
                st.markdown(f"[Download {lang} PDF]({output_path})")
            except Exception as e:
                print(f"❌ Failed to create PDF for {lang}: {e}")
                st.error(f"❌ Failed to create PDF for {lang}.")
        else:
            print(f"❌ Failed to translate to {lang}")
            st.error(f"❌ Failed to translate to {lang}.")
if __name__ == "__main__":
    run_translator_interface(pdf_path, selected_languages)
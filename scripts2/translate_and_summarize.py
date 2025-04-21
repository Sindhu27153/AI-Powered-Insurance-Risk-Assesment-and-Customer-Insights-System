import os
import tempfile
import shutil
import PyPDF2
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Frame, PageTemplate, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import inch
import streamlit as st
import re

# Language code to font path mapping
language_fonts = {
    "en": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSans-Regular.ttf",
    "ta": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansTamil-Regular.ttf",
    "hi": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansDevanagari-Regular.ttf",
    "te": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansTelugu-Regular.ttf",
    "kn": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansKannada-Regular.ttf",
    "ml": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansMalayalam-Regular.ttf",
    "bn": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansBengali-Regular.ttf",
    "gu": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansGujarati-Regular.ttf",
    "ur": "D:/AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System/fonts/NotoSansArabic-Regular.ttf"
}

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

# Translate text into target language
def translate_text(text, dest_language):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}")

# Summarize text
def summarize_text(text, language="english", sentences_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        raise RuntimeError(f"Error summarizing text: {e}")

# Generate aligned PDF using reportlab
def generate_translated_pdf(text, font_path, output_path, lang_title="Translated Insurance Policy"):
    try:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        pdfmetrics.registerFont(TTFont(font_name, font_path))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=60,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Justify',
            fontName=font_name,
            fontSize=12,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))

        elements = []

        title_style = ParagraphStyle(
            name='Title',
            fontName=font_name,
            fontSize=16,
            alignment=1,
            spaceAfter=20
        )
        elements.append(Paragraph(lang_title, title_style))

        # Clean and normalize text
        normalized_text = re.sub(r'\n+', '\n', text)  # Remove excessive line breaks
        normalized_text = re.sub(r'\s{2,}', ' ', normalized_text)  # Remove large white space
        paragraphs = [para.strip() for para in normalized_text.split('\n') if len(para.strip()) > 20]

        for para in paragraphs:
            block = KeepTogether([
                Paragraph(para, styles['Justify']),
                Spacer(1, 0.15 * inch)
            ])
            elements.append(block)

        doc.build(elements)

    except Exception as e:
        raise RuntimeError(f"Failed to generate aligned PDF: {e}")

# Main translator interface
def run_translator_interface(uploaded_file=None, selected_languages=None):
    st.title("📄 Policy Translator and Summarizer")
    st.markdown("Upload your insurance policy PDF and select the languages for translation and summarization.")

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
    }

    if uploaded_file and selected_languages:
        temp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(uploaded_file, out_file)

        try:
            extracted_text = extract_text_from_pdf(tmp_path)
            if not extracted_text.strip():
                st.warning("The uploaded PDF appears to be empty.")
                return
        except Exception as e:
            st.error(f"Failed to extract PDF text: {e}")
            return

        # English summary
        st.subheader("📝 Summary (English)")
        try:
            english_summary = summarize_text(extracted_text, language="english")
            st.success(english_summary)
        except Exception as e:
            st.error(f"Error summarizing English content: {e}")

        # Translate and summarize
        for lang_name in selected_languages:
            lang_code = language_map.get(lang_name)
            if lang_code:
                try:
                    translated_text = translate_text(extracted_text, lang_code)
                    font_path = language_fonts.get(lang_code, language_fonts["en"])

                    os.makedirs("Translated_Policies", exist_ok=True)
                    translated_pdf_path = os.path.join("Translated_Policies", f"Translated_Insurance_Policy_{lang_name}.pdf")

                    # Translate the summary
                    translated_summary = translate_text(english_summary, lang_code)

                    full_text = f"{translated_summary}\n\n{translated_text}"

                    generate_translated_pdf(full_text, font_path, translated_pdf_path, lang_title=f"Translated Insurance Policy ({lang_name})")

                    st.subheader(f"🌐 Translated Summary ({lang_name})")
                    st.write(translated_summary)

                    with open(translated_pdf_path, "rb") as f:
                        st.download_button(
                            label=f"📅 Download Translated Policy ({lang_name})",
                            data=f.read(),
                            file_name=f"Translated_Insurance_Policy_{lang_name}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error translating or saving PDF in {lang_name}: {e}")
            else:
                st.warning(f"❌ {lang_name} is not a valid language selection.")

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    else:
        if not uploaded_file:
            st.warning("Please upload an insurance policy PDF.")
        if not selected_languages:
            st.warning("Please select at least one language for translation.")

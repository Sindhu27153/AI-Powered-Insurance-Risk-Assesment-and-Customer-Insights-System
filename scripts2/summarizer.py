# summarizer.py
import PyPDF2
from googletrans import Translator
from transformers import pipeline

# Summarizer setup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()

def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])

def translate_to_english(text):
    return translator.translate(text, dest='en').text

def summarize_text(text):
    max_chunk = 1000
    summary = ""
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n"
    return summary

def translate_back(text, lang_code):
    return translator.translate(text, dest=lang_code).text

def run_summarizer_interface(pdf_path, lang_code):
    original_text = extract_pdf_text(pdf_path)
    english_text = translate_to_english(original_text)
    summary_en = summarize_text(english_text)
    summary_local = translate_back(summary_en, lang_code)
    return summary_local

if __name__ == "__main__":
    pdf_path = "path_to_your_pdf_file.pdf"
    lang_code = "en"  # For English; change this as needed for other languages

    # Now call the function with the provided arguments
    result = run_summarizer_interface(pdf_path, lang_code)
    
    # Print the summarized text in the target language
    print(result)
    run_summarizer_interface()
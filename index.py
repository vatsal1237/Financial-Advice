# === Imports ===
import fitz
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pdfplumber
def get_pdf_page_titles(pdf_path, max_lines=3):
    titles = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            title = " ".join(text.strip().split("\n")[:max_lines]) if text else "[No text on page]"
            titles.append((i + 1, title))
    return titles

# === Step 4: Gemini Setup ===
genai.configure(api_key="AIzaSyDNZVluiMytdMSd6LugnajKwOv34NBAeXc")  # Replace with your actual key
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Step 5: Load PDF and get relevant pages ===
pdf_path = "icici.pdf"
all_pages = extract_text_by_page(pdf_path)
titles = get_pdf_page_titles(pdf_path)
query = "give me the % to nav for ntpc in the ICICI Prudential Energy Opportunities Fund"

# Ask Gemini for relevant page numbers
gemini_response = gemini_model.generate_content(
    f"This is the query given by the user: {query}. Based on the following titles, select relevant page numbers (comma-separated) that are most related to the query:\n\n{titles}"
)

print("🧠 Gemini Page Numbers Response:", gemini_response.text)


# === Imports ===
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path

# === Step 1: Extract text by page ===
def extract_text_by_page(path):
    with fitz.open(path) as doc:
        return [page.get_text() for page in doc]

# === Step 2: Chunk text into overlapping segments ===
def chunk_text(text, max_words=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += max_words - overlap
    return chunks

# === Step 3: Get top lines from each page ===from pdf2image import convert_from_path
import google.generativeai as genai
from PIL import Image

# === Gemini Configuration ===
genai.configure(api_key="YOUR_API_KEY_HERE")  # Replace with your Gemini API key
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Function to extract page-wise summaries ===
def get_pdf_page_titles(pdf_path, first_page=None, last_page=None):
    """
    Returns a list of tuples: (page_number, Gemini-generated summary of that page).
    """
    # Convert selected or all pages to images
    if first_page and last_page:
        images = convert_from_path(pdf_path, dpi=300, first_page=first_page, last_page=last_page)
        page_offset = first_page - 1
    else:
        images = convert_from_path(pdf_path, dpi=300)
        page_offset = 0

    titles = []

    for i, image in enumerate(images):
        page_number = i + 1 + page_offset
        print(f"🧠 Processing Page {page_number}...")

        prompt = (
            "return the title and subtitle and two line summary of the page after reading it properly."
        )

        response = gemini_model.generate_content([image, prompt])
        titles.append((page_number, response.text.strip()))

    return titles


# === Step 4: Gemini Setup ===
genai.configure(api_key="AIzaSyDNZVluiMytdMSd6LugnajKwOv34NBAeXc")  # Replace with your actual key
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Step 5: Load PDF and get relevant pages ===
pdf_path = "icici.pdf"
all_pages = extract_text_by_page(pdf_path)
titles = get_pdf_page_titles(pdf_path)
query = "Summarise each fund’s annualised total return for the last 1, 3, 5 and 10 years, and show the gap versus its stated benchmark for every period."

# Ask Gemini for relevant page numbers
gemini_response = gemini_model.generate_content(
    f"This is the query given by the user: {query}. Based on the following summaries, select relevant page numbers comma-separated that are most related to the query there can be multiple pages upto 100 also if needed just comma seperate them:\n\n{titles}"
)

print("🧠 Gemini Page Numbers Response:", gemini_response.text)

# === Step 6: Extract and chunk only relevant pages ===
page_numbers = [int(p.strip()) for p in gemini_response.text.split(",") if p.strip().isdigit()]
relevant_text = " ".join([all_pages[p - 1] for p in page_numbers if p - 1 < len(all_pages)])
relevant_chunks = chunk_text(" ".join(relevant_text.split()))

# === Step 7: FAISS Index Creation ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(relevant_chunks, show_progress_bar=True)
embeddings_np = np.array(embeddings).astype("float32")
dim = embeddings_np.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings_np)

# === Step 8: Search Query with FAISS ===
def search_faiss(query, model, faiss_index, chunks, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

results = search_faiss(query, model, faiss_index, relevant_chunks, top_k=3)
context = " ".join(results)
#print("\n🔍 FAISS Top Results:\n", context)

# === Step 9: (Optional) Get Final Answer from Gemini ===
final_response = gemini_model.generate_content(
     f"The user asked: {query}. Based on the following context, provide a detailed and clear answer within 500 words:\n\n{context}"
)
print("\n📘 Gemini Final Answer:\n", final_response.text)

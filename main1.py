# === Imports ===
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pdfplumber

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

# === Step 3: Get top lines from each page ===
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
query = "Summarise each fund’s annualised total return for the last 1, 3, 5 and 10 years, and show the gap versus its stated benchmark for every period."

# Ask Gemini for relevant page numbers
gemini_response = gemini_model.generate_content(
    f"This is the query given by the user: {query}. Based on the following titles, select relevant page numbers comma-separated that are most related to the query:\n\n{titles}"
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

from pdf2image import convert_from_path
import google.generativeai as genai
from PIL import Image

# === Step 1: Convert first page of the PDF to image ===
images = convert_from_path("icici.pdf", dpi=300, first_page=20, last_page=20)
image = images[0]

# === Step 2: Configure Gemini API ===
genai.configure(api_key="AIzaSyDNZVluiMytdMSd6LugnajKwOv34NBAeXc")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-flash")

# === Step 3: Send image to Gemini and get analysis ===
response = model.generate_content(
    [image, "divide the pages using an imaginary grid based on white spaces and return a short summary of that page in a list"]
)

# === Step 4: Print Gemini's response ===
print("\n📄 Gemini Analysis of First Page:\n")
print(response.text)

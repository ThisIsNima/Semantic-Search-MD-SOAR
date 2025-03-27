import os
import json
import pandas as pd
import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = ""  # Change this to your directory with PDF files
OUTPUT_CSV = ""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Function to read and process PDF files
def read_pdf_files(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    if text.strip():  # Ensure text is not empty
                        documents[filename] = text
                    else:
                        print(f"Warning: No extractable text in {filename}")
            except PDFSyntaxError:
                print(f"Skipping invalid/corrupt PDF: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

# Read documents
documents = read_pdf_files(DATA_DIR)

# Vectorize documents
vectors = {doc: model.encode(text).tolist() for doc, text in documents.items()}

# Save to CSV
vector_data = [(doc, json.dumps(vec)) for doc, vec in vectors.items()]
df = pd.DataFrame(vector_data, columns=["filename", "vector"])
df.to_csv(OUTPUT_CSV, index=False)

# Save to JSON (alternative)
with open("vectors.json", "w", encoding="utf-8") as f:
    json.dump(vectors, f, indent=4)

print("PDF Vectorization complete!")

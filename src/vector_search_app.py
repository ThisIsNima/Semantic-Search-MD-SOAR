from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from flask_cors import CORS  # Enables Cross-Origin Resource Sharing for frontend access

# === Configuration ===
VECTOR_CSV = ""  # Path to your CSV with filename-vector pairs
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Pretrained embedding model

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow requests from frontend (e.g., Angular)

# === Load the Sentence Transformer Model ===
model = SentenceTransformer(MODEL_NAME)

# === Load and Prepare Vector Data ===
df = pd.read_csv(VECTOR_CSV)
df["vector"] = df["vector"].apply(lambda x: json.loads(x))  # Convert stored JSON strings to Python lists

# === Function for Cosine Similarity Search (for longer queries) ===
def find_closest_matches(query, top_n=5):
    query_vector = model.encode(query).tolist()  # Encode query into vector
    similarities = []

    for _, row in df.iterrows():
        similarity = 1 - cosine(query_vector, row["vector"])  # Cosine similarity (higher is better)
        similarities.append((row["filename"], similarity))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === Function for Hybrid Search (for shorter queries) ===
def hybrid_search(query, top_n=5):
    query_vector = model.encode(query).tolist()  # Encode query
    keywords = set(query.lower().split())  # Extract keywords from query

    results = []
    for _, row in df.iterrows():
        similarity = np.dot(query_vector, row["vector"])  # Dot product for fast similarity

        # Check if keywords match filename words (e.g., filename: 'climate_change_report.pdf')
        filename_words = set(row["filename"].lower().split("_"))
        keyword_match_score = len(keywords & filename_words) * 0.1  # Small boost per keyword match

        # Combine similarity and keyword match score
        results.append((row["filename"], similarity + keyword_match_score))

    # Sort by combined score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# === REST API Endpoint: /search ===
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()  # Get JSON data from request
    query = data.get("query", "")  # Extract query string

    # Return error if query is empty
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Choose search strategy based on query length
    word_count = len(query.split())
    if word_count > 5:
        results = find_closest_matches(query)
    else:
        results = hybrid_search(query)

    # Build response JSON with filenames and confidence scores
    response = [{"filename": filename, "confidence": round(score, 4)} for filename, score in results]
    return jsonify(response)

# === Run the App Locally ===
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for development

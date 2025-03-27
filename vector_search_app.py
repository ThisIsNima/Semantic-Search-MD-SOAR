from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from flask_cors import CORS

# === Configuration ===
VECTOR_CSV = "vectors.csv"
OUTPUT_CSV = "output.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app)

# === Load the Sentence Transformer Model ===
model = SentenceTransformer(MODEL_NAME)

# === Normalize helper ===
def normalize(text):
    return text.strip().lower()

# === Load and Prepare Vector Data ===
df = pd.read_csv(VECTOR_CSV)
df["vector"] = df["vector"].apply(lambda x: json.loads(x))
df["filename_normalized"] = df["filename"].apply(normalize)

# === Load UUID mapping from output.csv ===
uuid_df = pd.read_csv(OUTPUT_CSV)
uuid_df["title_normalized"] = uuid_df["title"].apply(normalize)
title_to_uuid = dict(zip(uuid_df["title_normalized"], uuid_df["item_uuid"]))

# === Function for Cosine Similarity Search (for longer queries) ===
def find_closest_matches(query, top_n=5):
    query_vector = model.encode(query).tolist()
    similarities = []

    for _, row in df.iterrows():
        similarity = 1 - cosine(query_vector, row["vector"])
        similarities.append((row["filename"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === Function for Hybrid Search (for shorter queries) ===
def hybrid_search(query, top_n=5):
    query_vector = model.encode(query).tolist()
    keywords = set(query.lower().split())

    results = []
    for _, row in df.iterrows():
        similarity = np.dot(query_vector, row["vector"])
        filename_words = set(row["filename"].lower().split("_"))
        keyword_match_score = len(keywords & filename_words) * 0.1
        results.append((row["filename"], similarity + keyword_match_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# === REST API Endpoint: /search ===
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    word_count = len(query.split())
    if word_count > 5:
        results = find_closest_matches(query)
    else:
        results = hybrid_search(query)

    # Enrich with item_uuid using normalized matching and fallback with underscores replaced by spaces
    response = []
    for filename, score in results:
        normalized_filename = normalize(filename)
        item_uuid = title_to_uuid.get(normalized_filename)

        if item_uuid is None:
            # Try again with underscores replaced by spaces
            alt_filename = filename.replace("_", " ")
            normalized_alt = normalize(alt_filename)
            item_uuid = title_to_uuid.get(normalized_alt)

        response.append({
            "filename": filename,
            "confidence": round(score, 4),
            "item_uuid": item_uuid
        })

    return jsonify(response)

# === Run the App Locally ===
if __name__ == "__main__":
    app.run(debug=True, port=5003)

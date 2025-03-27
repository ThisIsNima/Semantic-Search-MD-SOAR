import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Configuration
VECTOR_CSV = ""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load vectors
df = pd.read_csv(VECTOR_CSV)
df["vector"] = df["vector"].apply(lambda x: json.loads(x))  # Convert JSON strings to lists

# Function for regular cosine similarity search (for longer queries)
def find_closest_matches(query, top_n=5):
    query_vector = model.encode(query).tolist()
    
    similarities = []
    for _, row in df.iterrows():
        similarity = 1 - cosine(query_vector, row["vector"])  # Cosine similarity
        similarities.append((row["filename"], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

# Function for hybrid search (for short queries)
def hybrid_search(query, top_n=5):
    query_vector = model.encode(query).tolist()
    keywords = set(query.lower().split())

    results = []
    for _, row in df.iterrows():
        similarity = np.dot(query_vector, row["vector"])  # Dot product similarity

        # Boost similarity if filename contains keywords
        filename_words = set(row["filename"].lower().split("_"))
        keyword_match_score = len(keywords & filename_words) * 0.1  # Small boost for keyword match

        results.append((row["filename"], similarity + keyword_match_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# Interactive query loop
if __name__ == "__main__":
    while True:
        query_phrase = input("Enter a query phrase (or type 'exit' to quit): ")
        if query_phrase.lower() == "exit":
            print("Exiting program. Goodbye!")
            break
        
        # Choose search strategy based on query length
        word_count = len(query_phrase.split())
        if word_count > 5:
            results = find_closest_matches(query_phrase)
        else:
            results = hybrid_search(query_phrase)

        # Print results
        print("\nTop matches:")
        for filename, score in results:
            print(f"{filename}: Confidence: {score:.4f}")
        
        print("\nEnter your next query or type 'exit' to quit.")

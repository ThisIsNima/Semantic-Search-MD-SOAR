import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
# import psycopg2
# from psycopg2.extras import execute_values

# Configuration
DATA_DIR = ""  # Change this to your directory with text files.
OUTPUT_CSV = ""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "text_vectors"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Function to read and process text files
def read_text_files(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8", errors="ignore") as file:
                documents[filename] = file.read()
    return documents

# Read documents
documents = read_text_files(DATA_DIR)

# Vectorize documents
vectors = {doc: model.encode(text).tolist() for doc, text in documents.items()}

# Save to CSV
vector_data = [(doc, json.dumps(vec)) for doc, vec in vectors.items()]
df = pd.DataFrame(vector_data, columns=["filename", "vector"])
df.to_csv(OUTPUT_CSV, index=False)

# Save to JSON (alternative)
with open("vectors.json", "w", encoding="utf-8") as f:
    json.dump(vectors, f, indent=4)

# Upload to PostgreSQL (I will work on this part later)
    
def upload_to_postgres(db_config, table_name, data):
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        vector VECTOR(384)
    );
    '''
    cursor.execute(create_table_query)
    
    insert_query = f"INSERT INTO {table_name} (filename, vector) VALUES %s"
    execute_values(cursor, insert_query, data)
    
    conn.commit()
    cursor.close()
    conn.close()

# Example usage (replace with your credentials)
db_config = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

#upload_to_postgres(db_config, TABLE_NAME, vector_data)

print("Vectorization and upload complete!")

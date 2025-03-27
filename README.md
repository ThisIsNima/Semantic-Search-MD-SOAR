# 🔎 Semantic Search for MD-SOAR

This project provides a semantic search API for the [MD-SOAR](https://mdsoar.org/) repository. It uses sentence embeddings from the `sentence-transformers` library to allow natural language queries over pre-computed document vectors. The API supports both cosine similarity and a hybrid approach with keyword boosting for improved results on shorter queries.

## 🚀 Features

- RESTful API using Flask
- Embedding-based semantic similarity with Sentence Transformers
- Hybrid search strategy (vector + keyword boosting)
- Preprocessed document vectors for fast querying
- CORS enabled for Angular or other frontend integration

---
## 🧠 Requirements

- Python 3.12+
- Necessary pip packages:

```
pip install -e .
```


## 📆 Running the App

Make sure you have the `vectors.csv` and "output.csv" files
(see https://umd.app.box.com/folder/313765372458).

1) Download the "vectors.csv" and "output.csv" files from
   <https://umd.app.box.com/folder/313765372458> and place in
   the project root.

   **Note:** If different file names are used, change the `VECTOR_CSV` and
   `OUTPUT_CSV` variables in "src/vector_search_app.py".

2) Run the server:

```
python src/vector_search_app
```

## 🔍 API Usage

### Endpoint: `/search`
**Method:** `POST`
**Content-Type:** `application/json`

### Request Body:
```json
{
  "query": "climate change and environmental policy"
}
```

### Sample Response:
```json
[
  {
    "confidence": 0.7277,
    "filename": "Episode_5__The_Social_Science_of_the_Climate_Crisis_with_Dr._Tracey_Osborne.pdf.txt",
    "item_uuid": null
  },
  {
    "confidence": 0.7258,
    "filename": "Cities_worldwide_aren't_adapting_to_climate_change_quickly_enough.pdf.txt",
    "item_uuid": "2832c6fe-a51c-40ea-aee1-100d750daf35"
  },
  {
    "confidence": 0.6806,
    "filename": "Hybrid_Causality_Analysis_of_ENSO\u2019s_Global_Impacts_on_Climate_Variables_Based_on_Data-Driven_Analytics_and_Climate_Model_Simulation.pdf.txt",
    "item_uuid": "f7c0e645-cab0-4ac4-816a-b49dc2438d92"
  },
  ...
]
```

---

## 🌐 Frontend Integration

This API is CORS-enabled and can be directly integrated with an Angular or React frontend via HTTP POST requests.

```typescript
this.http.post<any[]>('http://localhost:5000/search', { query: this.queryText })
  .subscribe(results => {
    this.searchResults = results;
  });
```




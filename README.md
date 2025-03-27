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

- Python 3.7+
- Necessary pip packages


## 📆 Running the App

Make sure you have the `vectors.csv` file ready (contains `filename` and `vector` columns).



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
    "filename": "Climate_Change_Report.pdf",
    "confidence": 0.8765
  },
  {
    "filename": "Environmental_Policy_Summary.pdf",
    "confidence": 0.8491
  }
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




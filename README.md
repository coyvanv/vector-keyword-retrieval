# vector-keyword-retrieval
# 🧠 Hybrid BM25 + Embedding Retriever

This repository contains a hybrid document retrieval system that combines **BM25 keyword search** with **vector-based semantic search**. Documents are stored in a simple CSV format and can be queried using both textual and embedding-based input.

## 🔍 Use Cases

- Search relevant text chunks using both meaning (embeddings) and keywords.
- Build lightweight search tools without a full database.
- Integrate into chatbots, RAG pipelines, or document indexing workflows.
- Apply NLP on Dutch-language corpora (includes Dutch stopword handling).

## ⚙️ Requirements

Install the required packages:

```bash
pip install pandas numpy nltk rank_bm25
Also download NLTK stopwords once:

python
Copy
Edit
import nltk
nltk.download('stopwords')
```bash
📌 What You Need to Provide
To use this code:

📥 Document embeddings: You must generate your own embedding vectors (e.g. using OpenAI, HuggingFace, etc.).

🧾 Query embeddings: The query passed to retrieve_similar() must also be embedded beforehand.

📐 Consistent dimensions: All embeddings (document and query) must be the same length.

🚀 Example
python
Copy
Edit
retriever = CSVRetriever("data.csv")

retriever.add_document(
    content_id="doc1",
    chunk_id="1",
    embedding=[0.1, 0.2, 0.3],
    metadata={
        "title": "Cycling Policy",
        "chunk_content": "Amsterdam promotes cycling through new policies.",
        "municipality": "Amsterdam",
        "url": "https://example.com",
        "source": "official-docs"
    }
)

results = retriever.retrieve_similar(
    embedding=[0.1, 0.2, 0.3],
    query_text="bike infrastructure",
    top_k=3
)

for result in results:
    print(f"[{result.chunk_id}] {result.title} ({result.distance:.4f})")
    print(result.chunk_content)


# vector-keyword-retrieval
Overview
This project implements a hybrid document retrieval system using both BM25 (text-based relevance) and vector similarity (semantic matching via embeddings). It stores documents and their metadata in a CSV file and retrieves the most relevant chunks for a given query.

The CSVRetriever class combines the strengths of classical keyword-based retrieval and modern embedding-based similarity search to return high-quality matches.

Use Cases
Search across large CSV-based corpora without needing a full database.

Combine semantic search (via embeddings) and traditional keyword search (BM25).

Ideal for projects where you want flexible, explainable document retrieval.

Useful in NLP prototypes, document clustering, chatbots, or knowledge base assistants.

Requirements & Setup
Dependencies
Make sure to install required Python packages:

bash
Copy
Edit
pip install pandas numpy nltk rank_bm25
Also, ensure NLTK stopwords are available:

python
Copy
Edit
import nltk
nltk.download('stopwords')
What You Need to Provide
To use this system effectively, you must provide:

Embeddings
You are responsible for generating and supplying the embedding vector (as a list of floats) when:

Adding documents via add_document(...)

Querying via retrieve_similar(...)

Query Text (optional but recommended)
If you supply query_text, it will be used in BM25 search alongside the embedding-based search.

Consistent Embedding Length
All vectors (document and query) should have the same dimensionality.

Example Usage
python
Copy
Edit
retriever = CSVRetriever("docs.csv")

# Add document chunk
retriever.add_document(
    content_id="doc1",
    chunk_id="1",
    embedding=[0.1, 0.2, 0.3],
    metadata={
        "title": "Sample Doc",
        "chunk_content": "This is a test document about cycling in Amsterdam.",
        "url": "https://example.com",
        "municipality": "Amsterdam",
        "source": "test-source"
    }
)

# Retrieve top 5 relevant chunks
results = retriever.retrieve_similar(
    embedding=[0.1, 0.2, 0.3],  # same embedding size as document
    query_text="cycling in the city",
    top_k=5
)

for r in results:
    print(f"{r.title} ({r.distance:.4f}): {r.chunk_content}")

from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import csv
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('dutch')) #in this case I use Dutch stopwords
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    content_id: str
    chunk_id: str
    chunk_content: str
    title: str
    url: str
    distance: float
    municipality: str
    source: str
    cluster_name: str

class CSVRetriever:
    def __init__(self, csv_path: str, alpha: float = 0.5):
        self.csv_path = Path(csv_path)
        self.alpha = alpha
        self.bm25 = None
        self._ensure_csv_structure()

    def _ensure_csv_structure(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'content_id',
                    'chunk_id',
                    'embedding',
                    'title',
                    'url',
                    'source',
                    'municipality',
                    'state',
                    'chunk_content',
                    'cluster_name',
                    'embedding_strategy',
                    'embedding_model'
                ])

    def add_document(
        self,
        content_id: str,
        chunk_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add a document chunk to the CSV store and BM25 index."""
        try:
            new_row = {
                'content_id': content_id,
                'chunk_id': chunk_id,
                'embedding': ','.join(map(str, embedding)),
                'title': metadata.get('title', ''),
                'url': metadata.get('url', ''),
                'source': metadata.get('source', ''),
                'municipality': metadata.get('municipality', ''),
                'state': metadata.get('state', 'published'),
                'chunk_content': metadata.get('chunk_content', ''),
                'cluster_name': metadata.get('cluster_name', ''),
                'embedding_strategy': metadata.get('embedding_strategy', ''),
                'embedding_model': metadata.get('embedding_model', '')
            }
            # Append to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=new_row.keys())
                writer.writerow(new_row)
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def rebuild_bm25_index(self):
        self._initialize_bm25_index()

    def _initialize_bm25_index(self):
        df = pd.read_csv(self.csv_path)
        
        if df.empty:
            logger.warning("CSV file is empty. No documents to index.")
            return

        # Convert chunk content to lowercase before indexing
        chunks = [{"text": row['chunk_content'].lower(), "metadata": {}} for _, row in df.iterrows()]

        logger.info(f"Initializing BM25 index with {len(chunks)} chunks.")
        self.bm25 = create_bm25_index(chunks)
        logger.info("BM25 index initialized successfully.")

    def _clean_query(self, query: str) -> str:
        """Remove punctuation and clean the query text."""
        query = query.lower()
        translator = str.maketrans('', '', string.punctuation)
        cleaned_query = query.translate(translator)
        cleaned_query = ' '.join(cleaned_query.split())
        return cleaned_query

    def retrieve_similar(
        self,
        embedding: List[float],
        query_text: Optional[str] = None,
        municipality: Optional[str] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve similar chunks using fusion of vector-based and BM25 retrieval."""
        try:
            if self.bm25 is None:
                print("BM25-index not initialized, initializing...")
                self.rebuild_bm25_index()


            if query_text is not None:
                original_query = query_text
                query_text = self._clean_query(query_text)
            vector_scores_accumulator = {}
            bm25_scores_accumulator = {}
            combined_scores_accumulator = {}

            # Run retrieval 10 times
            for i in range(10):
                
                # Vector-based retrieval
                df = pd.read_csv(self.csv_path)
                df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x, sep=',').astype(np.float32))
                query_embedding = np.array(embedding).astype(np.float32)
                df['vector_score'] = df['embedding'].apply(lambda x: self._cosine_similarity(x, query_embedding))

                # BM25 retrieval
                if query_text is not None and self.bm25 is not None:
                    # Convert query tokens to lowercase
                    query_tokens = [word.lower() for word in query_text.split() if word.lower() not in stop_words]
                    # Convert chunk content to lowercase
                    chunks = [{"text": row['chunk_content'].lower(), "metadata": {}} for _, row in df.iterrows()]
                    bm25_results = bm25_search(self.bm25, chunks, query_text, len(df))
                    bm25_scores = {result['metadata']['index']: result['bm25_score'] for result in bm25_results}
                    df['bm25_score'] = df.index.map(lambda i: bm25_scores.get(i, 0))
                else:
                    df['bm25_score'] = 0.0

                # Combine scores
                df['combined_score'] = self.alpha * df['vector_score'] + (1 - self.alpha) * df['bm25_score']

                # Accumulate scores
                for _, row in df.iterrows():
                    chunk_id = row['chunk_id']
                    if chunk_id not in vector_scores_accumulator:
                        vector_scores_accumulator[chunk_id] = []
                        bm25_scores_accumulator[chunk_id] = []
                        combined_scores_accumulator[chunk_id] = []
                    
                    vector_scores_accumulator[chunk_id].append(row['vector_score'])
                    bm25_scores_accumulator[chunk_id].append(row['bm25_score'])
                    combined_scores_accumulator[chunk_id].append(row['combined_score'])

            # Calculate average scores
            df = pd.read_csv(self.csv_path)
            df['avg_vector_score'] = df['chunk_id'].map(lambda x: np.mean(vector_scores_accumulator.get(x, [0])))
            df['avg_bm25_score'] = df['chunk_id'].map(lambda x: np.mean(bm25_scores_accumulator.get(x, [0])))
            df['avg_combined_score'] = df['chunk_id'].map(lambda x: np.mean(combined_scores_accumulator.get(x, [0])))

            # Create a list to store all results
            results_data = []
            
            # Collect data for each chunk
            for _, row in df.iterrows():
                chunk_id = row['chunk_id']
                chunk_data = {
                    'Chunk ID': chunk_id,
                    'Vector Score': f"{row['avg_vector_score']:.4f}",
                    'BM25 Score': f"{row['avg_bm25_score']:.4f}",
                    'Combined Score': f"{row['avg_combined_score']:.4f}"
                }
                results_data.append(chunk_data)

            # Create DataFrame and display as table
            results_df = pd.DataFrame(results_data)
            print(results_df.to_string(index=False))

            # Sort by average combined score and get top k results
            results = df.sort_values('avg_combined_score', ascending=False).head(top_k)

            # Return results
            return [
                RetrievalResult(
                    content_id=row['content_id'],
                    chunk_id=row['chunk_id'],
                    chunk_content=row['chunk_content'],
                    title=row['title'],
                    url=row['url'],
                    distance=row['avg_combined_score'],
                    municipality=row['municipality'],
                    source=row['source'],
                    cluster_name=row['cluster_name']
                ) for _, row in results.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _get_top_keywords(bm25, doc_index: int, query_tokens: List[str]) -> List[str]:
        """
        Get the top keywords for a document based on BM25 scores.
        """
        # Get term frequencies for the document
        term_scores = {}
        for term in query_tokens:
            # Get the actual term frequency in the document
            doc_term_freq = bm25.doc_freqs[doc_index].get(term, 0)
            term_scores[term] = doc_term_freq
        
        # Sort the terms by frequency
        top_terms = sorted(term_scores.keys(), key=lambda x: term_scores[x], reverse=True)
        return top_terms

    @staticmethod
    def _get_most_common_keywords(keywords_lists):
        """Get the most common keywords from multiple runs."""
        if not keywords_lists:
            return []
        
        # Flatten the list of lists
        all_keywords = [keyword for sublist in keywords_lists for keyword in sublist]
        
        # Count occurrences
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by count and get top 5
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:5]  # Return tuples of (keyword, count)

def create_bm25_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    tokenized_docs = [text.split() for text in texts]
    

    bm25 = BM25Okapi(tokenized_docs)
    

    
    return bm25

def bm25_search(bm25, chunks, query, k=5):
    """
    Search the BM25 index with a query.
    
    Args:
        bm25 (BM25Okapi): BM25 index
        chunks (List[Dict]): List of text chunks
        query (str): Query string
        k (int): Number of results to return
        
    Returns:
        List[Dict]: Top k results with scores
    """
    # Tokenize the query by splitting it into individual words
    query_tokens = query.split()
    
    # Get BM25 scores for the query tokens against the indexed documents
    scores = bm25.get_scores(query_tokens)
    
    # Initialize an empty list to store results with their scores
    results = []
    
    # Iterate over the scores and corresponding chunks
    for i, score in enumerate(scores):
        # Create a copy of the metadata to avoid modifying the original
        metadata = chunks[i].get("metadata", {}).copy()
        # Add index to metadata
        metadata["index"] = i
        
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # Add metadata with index
            "bm25_score": float(score)
        })
    
    # Sort the results by BM25 score in descending order
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    # Return the top k results
    return results[:k]
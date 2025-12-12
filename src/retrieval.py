"""
Hybrid Retrieval Module

This module implements hybrid retrieval combining BM25 and dense vector search.
"""

import json
import pickle
import os
import logging
import requests
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)
from .config import RetrievalConfig
from .vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


def is_api_error(exception):
    """Check if exception is an API error that should be retried"""
    error_str = str(exception).lower()
    return any([
        '429' in error_str,
        'rate limit' in error_str,
        'quota' in error_str,
        'resource exhausted' in error_str,
        'resource_exhausted' in error_str,
        'too many requests' in error_str,
        'service unavailable' in error_str,
        '503' in error_str,
        '500' in error_str,
        'internal server error' in error_str,
        'connection' in error_str,
        'timeout' in error_str,
    ])


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense vector search"""

    def __init__(self, config: RetrievalConfig, vector_store: VectorStore):
        """
        Initialize the hybrid retriever

        Args:
            config: Retrieval configuration
            vector_store: Vector store for dense retrieval
        """
        self.config = config
        self.vector_store = vector_store
        self.bm25_retrievers = {}  # Collection name -> BM25 retriever
        self.corpus_data = {}      # Collection name -> corpus documents
        self.bm25_index_dir = "bm25_indexes"  # Directory to save BM25 indexes

        # Load Jina API key from environment
        self.jina_api_key = os.getenv("JINA_API_KEY")
        if self.jina_api_key:
            print("✓ Jina API key loaded for reranking")
        else:
            print("⚠ Jina API key not found, will use fallback reranking")

        # Create directory for BM25 indexes if it doesn't exist
        os.makedirs(self.bm25_index_dir, exist_ok=True)

    def _get_bm25_index_path(self, collection_name: str) -> str:
        """Get path to BM25 index file for a collection"""
        return os.path.join(self.bm25_index_dir, f"{collection_name}_bm25.pkl")

    def _get_corpus_path(self, collection_name: str) -> str:
        """Get path to corpus data file for a collection"""
        return os.path.join(self.bm25_index_dir, f"{collection_name}_corpus.pkl")

    def _save_bm25_index(self, collection_name: str):
        """Save BM25 index and corpus data to disk"""
        index_path = self._get_bm25_index_path(collection_name)
        corpus_path = self._get_corpus_path(collection_name)

        with open(index_path, 'wb') as f:
            pickle.dump(self.bm25_retrievers[collection_name], f)

        with open(corpus_path, 'wb') as f:
            pickle.dump(self.corpus_data[collection_name], f)

        print(f"✓ BM25 index saved to {index_path}")

    def _load_bm25_index(self, collection_name: str) -> bool:
        """Load BM25 index and corpus data from disk"""
        index_path = self._get_bm25_index_path(collection_name)
        corpus_path = self._get_corpus_path(collection_name)

        if not os.path.exists(index_path) or not os.path.exists(corpus_path):
            return False

        try:
            with open(index_path, 'rb') as f:
                self.bm25_retrievers[collection_name] = pickle.load(f)

            with open(corpus_path, 'rb') as f:
                self.corpus_data[collection_name] = pickle.load(f)

            print(f"✓ BM25 index loaded from {index_path}")
            return True
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def index_corpus_bm25(self, collection_name: str, corpus_file: str, force_reindex: bool = False) -> bool:
        """
        Index corpus for BM25 retrieval with persistence

        Args:
            collection_name: Name of the collection
            corpus_file: Path to corpus JSONL file
            force_reindex: Force reindexing even if index exists

        Returns:
            True if indexing was successful
        """
        # Try to load existing index if not forcing reindex
        if not force_reindex and self._load_bm25_index(collection_name):
            num_docs = len(self.corpus_data[collection_name])
            print(f"Using existing BM25 index ({num_docs} documents)")
            return True

        print(f"Loading corpus for BM25 indexing from {corpus_file}...")

        # Load documents
        corpus = []
        corpus_ids = []

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                corpus.append(doc)
                corpus_ids.append(doc['_id'])

        print(f"Loaded {len(corpus)} documents for BM25")

        # Store corpus data for retrieval
        self.corpus_data[collection_name] = corpus

        # Tokenize corpus for BM25
        print("Building BM25 index...")
        tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        self.bm25_retrievers[collection_name] = BM25Okapi(tokenized_corpus)
        print("BM25 indexing complete!")

        # Save index to disk
        self._save_bm25_index(collection_name)

        return True
    
    def retrieve_bm25(self, collection_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25

        Args:
            collection_name: Collection to search
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        # Auto-load BM25 index if not in memory but exists on disk
        if collection_name not in self.bm25_retrievers:
            if self._load_bm25_index(collection_name):
                print(f"Auto-loaded BM25 index for {collection_name}")
            else:
                print(f"BM25 retriever not available for collection {collection_name}")
                return []
        
        bm25 = self.bm25_retrievers[collection_name]
        corpus = self.corpus_data[collection_name]
        
        # Tokenize query
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            doc = corpus[idx]
            results.append({
                'document_id': doc['_id'],
                'text': doc['text'],
                'title': doc.get('title', ''),
                'score': float(scores[idx]),
                'source': 'bm25',
                'metadata': doc.get('metadata', {})
            })
        
        return results
    
    def retrieve_dense(self, collection_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using dense vector search
        
        Args:
            collection_name: Collection to search
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        results = self.vector_store.search(collection_name, query, top_k)
        
        # Add source information
        for result in results:
            result['source'] = 'dense'
            
        return results
    
    def hybrid_retrieve(self, collection_name: str, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for multiple query variants
        
        Args:
            collection_name: Collection to search
            queries: List of query variants
            
        Returns:
            List of unique retrieved documents
        """
        all_candidates = []
        
        for query in queries:
            # BM25 retrieval
            bm25_results = self.retrieve_bm25(
                collection_name, query, self.config.bm25_top_k
            )
            all_candidates.extend(bm25_results)
            
            # Dense retrieval
            dense_results = self.retrieve_dense(
                collection_name, query, self.config.dense_top_k
            )
            all_candidates.extend(dense_results)
        
        # Deduplicate by document_id
        seen_ids = set()
        unique_candidates = []
        
        for doc in all_candidates:
            doc_id = doc['document_id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_candidates.append(doc)
        
        print(f"Retrieved {len(unique_candidates)} unique candidates from {len(all_candidates)} total results")
        
        # Limit to max candidates
        if len(unique_candidates) > self.config.max_candidates:
            unique_candidates = unique_candidates[:self.config.max_candidates]
            print(f"Limited to {self.config.max_candidates} candidates")
        
        return unique_candidates

    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _rerank_with_jina(self, query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Jina AI reranking API

        Args:
            query: Query text
            candidates: List of candidate documents
            top_n: Number of top documents to return

        Returns:
            Reranked list of documents

        Raises:
            Exception: If API call fails after retries
        """
        if not self.jina_api_key:
            raise ValueError("Jina API key not configured")

        if not candidates:
            return []

        # Prepare documents for Jina API
        documents = [doc['text'] for doc in candidates]

        # Prepare API request
        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        data = {
            "model": "jina-reranker-v3",
            "query": query,
            "top_n": min(top_n, len(documents)),
            "documents": documents,
            "return_documents": False
        }

        # Make API call
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            # Process results
            reranked_docs = []
            for item in result.get('results', []):
                idx = item['index']
                score = item['relevance_score']

                # Get original document and update score
                doc = candidates[idx].copy()
                doc['rerank_score'] = score
                doc['final_score'] = score
                reranked_docs.append(doc)

            return reranked_docs

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Jina reranking API: {e}")
            raise

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Jina AI API or fallback to simple scoring

        Args:
            query: Original query
            candidates: List of candidate documents

        Returns:
            Reranked list of documents
        """
        if not candidates:
            return []

        # Try to use Jina reranking API if available
        if self.jina_api_key:
            try:
                print("Using Jina AI reranking...")
                reranked = self._rerank_with_jina(query, candidates, self.config.final_top_k)
                print(f"✓ Jina reranking complete ({len(reranked)} documents)")
                return reranked
            except Exception as e:
                logger.warning(f"Jina reranking failed, falling back to simple reranking: {e}")
                print(f"⚠ Jina reranking failed, using fallback method")

        # Fallback: Simple reranking by normalizing and combining BM25 and dense scores
        print("Using fallback reranking (score normalization)...")
        bm25_docs = [doc for doc in candidates if doc.get('source') == 'bm25']
        dense_docs = [doc for doc in candidates if doc.get('source') == 'dense']

        # Normalize scores within each group
        if bm25_docs:
            max_bm25_score = max(doc['score'] for doc in bm25_docs)
            if max_bm25_score > 0:
                for doc in bm25_docs:
                    doc['normalized_score'] = doc['score'] / max_bm25_score
            else:
                for doc in bm25_docs:
                    doc['normalized_score'] = 0.0

        if dense_docs:
            max_dense_score = max(doc['score'] for doc in dense_docs)
            if max_dense_score > 0:
                for doc in dense_docs:
                    doc['normalized_score'] = doc['score'] / max_dense_score
            else:
                for doc in dense_docs:
                    doc['normalized_score'] = 0.0

        # Combine scores (simple average)
        for doc in candidates:
            doc['final_score'] = doc.get('normalized_score', doc['score'])

        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)

        # Return top_k
        return reranked[:self.config.final_top_k]
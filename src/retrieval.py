"""
Hybrid Retrieval Module

This module implements hybrid retrieval combining BM25 and dense vector search.
"""

import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from .config import RetrievalConfig
from .vector_store import VectorStore


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
        
    def index_corpus_bm25(self, collection_name: str, corpus_file: str) -> bool:
        """
        Index corpus for BM25 retrieval
        
        Args:
            collection_name: Name of the collection
            corpus_file: Path to corpus JSONL file
            
        Returns:
            True if indexing was successful
        """
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
        if collection_name not in self.bm25_retrievers:
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
    
    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using a simple scoring strategy
        
        Args:
            query: Original query
            candidates: List of candidate documents
            
        Returns:
            Reranked list of documents
        """
        # Simple reranking: normalize and combine BM25 and dense scores
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
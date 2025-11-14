"""
Embedding Module for Multi-Turn RAG System

This module handles text embedding generation using dependency injection.
"""

from typing import List
from .config import EmbeddingConfig
from .embedding_providers import EmbeddingProvider, create_embedding_provider


class EmbeddingService:
    """Service for generating text embeddings using dependency injection"""
    
    def __init__(self, config: EmbeddingConfig, provider: EmbeddingProvider = None):
        """
        Initialize the embedding service
        
        Args:
            config: Configuration for the embedding model
            provider: Optional embedding provider (if None, will create based on config)
        """
        self.config = config
        self.provider = provider or create_embedding_provider(config)
        
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text to embed
            task_type: Task type for embedding (defaults to config task_type)
            
        Returns:
            List of float values representing the embedding
        """
        return self.provider.embed_text(text, task_type)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.provider.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query text
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        return self.provider.embed_query(query)
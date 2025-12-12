"""
Embedding Providers Module

This module implements different embedding providers using dependency injection.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log
)
import logging
from .config import EmbeddingConfig

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


def is_rate_limit_error(exception):
    """Check if exception is a rate limit or quota error"""
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
    ])


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """Embed a single text"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query text"""
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Gemini embedding provider"""
        self.config = config
        
        from google import genai
        from google.genai import types
        
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=config.api_key or os.getenv("GEMINI_API_KEY"))
    
    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """Embed text using Gemini with retry on rate limits"""
        if task_type is None:
            task_type = self.config.task_type

        embedding = self.client.models.embed_content(
            model=self.config.model_id,
            contents=text,
            config=self.types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.config.dimension
            )
        )
        return embedding.embeddings[0].values
    
    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using batch processing with retry on rate limits"""
        # Gemini API supports batch embedding
        result = self.client.models.embed_content(
            model=self.config.model_id,
            contents=texts,
            config=self.types.EmbedContentConfig(
                task_type="retrieval_document",
                output_dimensionality=self.config.dimension
            )
        )
        return [emb.values for emb in result.embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query text"""
        return self.embed_text(query, task_type="retrieval_query")


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize HuggingFace embedding provider"""
        self.config = config
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.model_id)
        except ImportError:
            raise ImportError("sentence-transformers is required for HuggingFace embeddings. Install with: pip install sentence-transformers")
    
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """Embed text using HuggingFace model"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using batch processing"""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query text"""
        return self.embed_text(query)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedding provider"""
        self.config = config
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai is required for OpenAI embeddings. Install with: pip install openai")
    
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """Embed text using OpenAI"""
        response = self.client.embeddings.create(
            model=self.config.model_id,
            input=text
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        # OpenAI supports batch processing
        response = self.client.embeddings.create(
            model=self.config.model_id,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query text"""
        return self.embed_text(query)


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """
    Factory function to create embedding provider based on configuration
    
    Args:
        config: Embedding configuration
        
    Returns:
        Embedding provider instance
    """
    if config.provider == "gemini":
        return GeminiEmbeddingProvider(config)
    elif config.provider == "huggingface":
        return HuggingFaceEmbeddingProvider(config)
    elif config.provider == "openai":
        return OpenAIEmbeddingProvider(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
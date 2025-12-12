"""
Configuration Module for Multi-Turn RAG System

This module contains all configuration parameters for the system.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_id: str = "gemini-embedding-001"
    dimension: int = 768
    task_type: str = "retrieval_document"
    api_key: Optional[str] = None
    provider: str = "gemini"  # gemini, huggingface, openai
    batch_size: int = 100  # Batch size for embedding multiple documents
    
    
@dataclass
class TextModelConfig:
    """Configuration for text generation models"""
    model_id: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.9
    api_key: Optional[str] = None
    provider: str = "gemini"  # gemini, openai, anthropic, etc.


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database"""
    url: Optional[str] = None
    api_key: Optional[str] = None
    distance_metric: str = "COSINE"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters"""
    bm25_top_k: int = 10
    dense_top_k: int = 10
    final_top_k: int = 5
    max_candidates: int = 50


@dataclass
class QueryDiversificationConfig:
    """Configuration for query diversification"""
    num_variants: int = 5
    use_entity_focus: bool = True
    use_action_focus: bool = True
    use_paraphrase: bool = True
    use_relation_focus: bool = True


@dataclass
class SystemConfig:
    """Main configuration class for the multi-turn RAG system"""
    embedding: EmbeddingConfig
    text_model: TextModelConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    query_diversification: QueryDiversificationConfig
    
    def __init__(self):
        self.embedding = EmbeddingConfig(
            model_id=os.getenv("EMBEDDING_MODEL_ID", "gemini-embedding-001"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            api_key=os.getenv("GEMINI_API_KEY"),
            provider=os.getenv("EMBEDDING_PROVIDER", "gemini")
        )
        self.text_model = TextModelConfig(
            model_id=os.getenv("TEXT_MODEL_ID", "gemini-2.5-flash"),
            temperature=float(os.getenv("TEXT_MODEL_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("TEXT_MODEL_MAX_TOKENS", "1024")),
            top_p=float(os.getenv("TEXT_MODEL_TOP_P", "0.9")),
            api_key=os.getenv("GEMINI_API_KEY"),
            provider=os.getenv("TEXT_MODEL_PROVIDER", "gemini")
        )
        self.qdrant = QdrantConfig(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.retrieval = RetrievalConfig(
            bm25_top_k=int(os.getenv("BM25_TOP_K", "10")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "10")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
            max_candidates=int(os.getenv("MAX_CANDIDATES", "50"))
        )
        self.query_diversification = QueryDiversificationConfig(
            num_variants=int(os.getenv("QUERY_VARIANTS", "5"))
        )


def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        yaml_file: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def create_config_from_yaml(yaml_config: Dict[str, Any]) -> SystemConfig:
    """
    Create SystemConfig from YAML configuration
    
    Args:
        yaml_config: Dictionary with YAML configuration
        
    Returns:
        SystemConfig instance
    """
    # Start with default values from environment
    config = SystemConfig()
    
    # Override with YAML values
    if 'embedding' in yaml_config:
        emb_config = yaml_config['embedding']
        if 'provider' in emb_config:
            config.embedding.provider = emb_config['provider']
        if 'model_id' in emb_config:
            config.embedding.model_id = emb_config['model_id']
        if 'dimension' in emb_config:
            config.embedding.dimension = emb_config['dimension']
        if 'task_type' in emb_config:
            config.embedding.task_type = emb_config['task_type']
    
    if 'text_model' in yaml_config:
        text_config = yaml_config['text_model']
        if 'provider' in text_config:
            config.text_model.provider = text_config['provider']
        if 'model_id' in text_config:
            config.text_model.model_id = text_config['model_id']
        if 'temperature' in text_config:
            config.text_model.temperature = text_config['temperature']
        if 'max_tokens' in text_config:
            config.text_model.max_tokens = text_config['max_tokens']
        if 'top_p' in text_config:
            config.text_model.top_p = text_config['top_p']
    
    if 'retrieval' in yaml_config:
        ret_config = yaml_config['retrieval']
        if 'bm25_top_k' in ret_config:
            config.retrieval.bm25_top_k = ret_config['bm25_top_k']
        if 'dense_top_k' in ret_config:
            config.retrieval.dense_top_k = ret_config['dense_top_k']
        if 'final_top_k' in ret_config:
            config.retrieval.final_top_k = ret_config['final_top_k']
        if 'max_candidates' in ret_config:
            config.retrieval.max_candidates = ret_config['max_candidates']
    
    if 'query_diversification' in yaml_config:
        qd_config = yaml_config['query_diversification']
        if 'num_variants' in qd_config:
            config.query_diversification.num_variants = qd_config['num_variants']
        if 'use_entity_focus' in qd_config:
            config.query_diversification.use_entity_focus = qd_config['use_entity_focus']
        if 'use_action_focus' in qd_config:
            config.query_diversification.use_action_focus = qd_config['use_action_focus']
        if 'use_paraphrase' in qd_config:
            config.query_diversification.use_paraphrase = qd_config['use_paraphrase']
        if 'use_relation_focus' in qd_config:
            config.query_diversification.use_relation_focus = qd_config['use_relation_focus']
    
    return config


def load_config(yaml_file: str = None) -> SystemConfig:
    """
    Load configuration from environment variables and optional YAML file
    
    Args:
        yaml_file: Optional path to YAML configuration file
        
    Returns:
        SystemConfig instance
    """
    if yaml_file:
        yaml_config = load_yaml_config(yaml_file)
        return create_config_from_yaml(yaml_config)
    else:
        return SystemConfig()
"""
Multi-Turn Conversational RAG Pipeline

This package implements the multi-turn conversational retrieval-augmented 
generation system as described in the architecture document.
"""

from .config import SystemConfig, load_config
from .pipeline import Pipeline, create_pipeline

__version__ = "1.0.0"
__all__ = ["SystemConfig", "load_config", "Pipeline", "create_pipeline"]
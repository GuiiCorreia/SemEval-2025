"""
DSPy Integrated RAG Pipeline

This module provides a unified pipeline that integrates:
1. Multi-hop retrieval (or standard retrieval)
2. Answerability classification
3. Response generation

The pipeline produces output compatible with the evaluation framework.
"""

import os
import dspy
from typing import List, Dict, Any, Optional

from src.dspy_multihop import MultiHopRetriever, create_multihop_retriever
from src.dspy_classification import (
    AnswerabilityClassifier,
    ResponseGenerator,
    format_conversation_history,
    format_documents
)


class IntegratedRAGPipeline:
    """
    Integrated RAG pipeline combining retrieval, classification, and generation.

    Flow:
    1. Multi-hop retrieval -> retrieved documents
    2. Classification -> ANSWERABLE/UNANSWERABLE/CONVERSATIONAL/PARTIAL
    3. Response generation -> final response based on classification
    """

    def __init__(
        self,
        hybrid_retriever,
        collection_name: str = None,
        num_hops: int = 3,
        classification_lm: dspy.LM = None,
        generation_lm: dspy.LM = None,
        api_key: str = None
    ):
        """
        Initialize the integrated pipeline.

        Args:
            hybrid_retriever: HybridRetriever instance for document retrieval
            collection_name: Default collection name
            num_hops: Number of retrieval hops (default: 3)
            classification_lm: LM for classification (default: gemini-3-flash-preview)
            generation_lm: LM for generation (default: gemini-3-pro-preview)
            api_key: API key for creating default LMs
        """
        # Retrieval component
        self.retriever = create_multihop_retriever(
            hybrid_retriever=hybrid_retriever,
            collection_name=collection_name,
            num_hops=num_hops
        )

        # Classification component
        self.classifier = AnswerabilityClassifier()

        # Generation component
        self.generator = ResponseGenerator()

        # Configure LMs
        if classification_lm is None and api_key:
            classification_lm = dspy.LM(
                "gemini/gemini-3-flash-preview",
                api_key=api_key,
                temperature=0.2,
                max_tokens=2000,
                cache=False,
            )

        if generation_lm is None and api_key:
            generation_lm = dspy.LM(
                "gemini/gemini-3-pro-preview",
                api_key=api_key,
                temperature=0.4,
                max_tokens=10000,
                cache=False,
            )

        self.classification_lm = classification_lm
        self.generation_lm = generation_lm

    def process_query(
        self,
        current_question: str,
        conversation_history: List[Dict[str, str]] = None,
        collection_name: str = None,
        retrieval_only: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query through the full pipeline.

        Args:
            current_question: The user's current question
            conversation_history: Previous conversation turns
            collection_name: Collection to search (overrides default)
            retrieval_only: If True, skip classification and generation

        Returns:
            Dictionary with pipeline results in evaluation-compatible format
        """
        conversation_history = conversation_history or []

        # Step 1: Multi-hop retrieval
        print("\n[1/3] Running multi-hop retrieval...")
        retrieval_result = self.retriever.process_query(
            current_question=current_question,
            conversation_history=conversation_history,
            collection_name=collection_name,
            retrieval_only=True
        )

        retrieved_docs = retrieval_result.get('pipeline_metadata', {}).get('retrieved_documents', [])
        print(f"  Retrieved {len(retrieved_docs)} documents")

        if retrieval_only:
            return {
                'response_text': '',
                'classification': '',
                'pipeline_metadata': retrieval_result.get('pipeline_metadata', {})
            }

        # Format inputs for classification and generation
        conv_history_str = format_conversation_history(conversation_history)
        docs_str = format_documents(retrieved_docs)

        # Step 2: Classification
        print("[2/3] Classifying answerability...")
        if self.classification_lm:
            with dspy.context(lm=self.classification_lm):
                classification_result = self.classifier(
                    conversation_history=conv_history_str,
                    current_question=current_question,
                    retrieved_documents=docs_str
                )
        else:
            classification_result = self.classifier(
                conversation_history=conv_history_str,
                current_question=current_question,
                retrieved_documents=docs_str
            )

        classification = classification_result.classification
        print(f"  Classification: {classification}")

        # Step 3: Response generation
        print("[3/3] Generating response...")
        if self.generation_lm:
            with dspy.context(lm=self.generation_lm):
                generation_result = self.generator(
                    conversation_history=conv_history_str,
                    current_question=current_question,
                    retrieved_documents=docs_str,
                    classification=classification
                )
        else:
            generation_result = self.generator(
                conversation_history=conv_history_str,
                current_question=current_question,
                retrieved_documents=docs_str,
                classification=classification
            )

        response_text = generation_result.response
        print(f"  Response generated ({len(response_text)} chars)")

        # Build pipeline metadata
        pipeline_metadata = retrieval_result.get('pipeline_metadata', {})
        pipeline_metadata['classification'] = classification
        pipeline_metadata['classification_reasoning'] = getattr(classification_result, 'reasoning', '')

        return {
            'response_text': response_text,
            'classification': classification,
            'pipeline_metadata': pipeline_metadata
        }

    def load_optimized_modules(
        self,
        retriever_path: str = None,
        classifier_path: str = None,
        generator_path: str = None
    ):
        """
        Load optimized retriever, classifier, and generator modules.

        Args:
            retriever_path: Path to optimized retriever weights
            classifier_path: Path to optimized classifier weights
            generator_path: Path to optimized generator weights
        """
        if retriever_path and os.path.exists(retriever_path):
            self.retriever.load(retriever_path)
            print(f"Loaded optimized retriever from {retriever_path}")

        if classifier_path and os.path.exists(classifier_path):
            self.classifier.load(classifier_path)
            print(f"Loaded optimized classifier from {classifier_path}")

        if generator_path and os.path.exists(generator_path):
            self.generator.load(generator_path)
            print(f"Loaded optimized generator from {generator_path}")


def create_integrated_pipeline(
    hybrid_retriever,
    collection_name: str = None,
    num_hops: int = 3,
    api_key: str = None,
    retriever_path: str = None,
    classifier_path: str = None,
    generator_path: str = None
) -> IntegratedRAGPipeline:
    """
    Factory function to create an integrated RAG pipeline.

    Args:
        hybrid_retriever: HybridRetriever instance
        collection_name: Default collection name
        num_hops: Number of retrieval hops
        api_key: API key for Gemini
        retriever_path: Path to optimized retriever (optional)
        classifier_path: Path to optimized classifier (optional)
        generator_path: Path to optimized generator (optional)

    Returns:
        Configured IntegratedRAGPipeline
    """
    pipeline = IntegratedRAGPipeline(
        hybrid_retriever=hybrid_retriever,
        collection_name=collection_name,
        num_hops=num_hops,
        api_key=api_key
    )

    # Load optimized modules if paths provided
    pipeline.load_optimized_modules(
        retriever_path=retriever_path,
        classifier_path=classifier_path,
        generator_path=generator_path
    )

    return pipeline
"""
Query Reformulation Module

This module handles context-aware query reformulation and query diversification.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)
from .config import QueryDiversificationConfig, TextModelConfig

# Load environment variables
load_dotenv()

# Setup logging
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
    ])


class QueryReformulator:
    """Handles query reformulation using Chain-of-Thought reasoning"""
    
    def __init__(self, config: TextModelConfig = None):
        """Initialize the query reformulator"""
        self.config = config or TextModelConfig(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.client = genai.Client(api_key=self.config.api_key)
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def cot_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Rewrite query using contextual understanding optimized for hybrid retrieval

        Args:
            query: Current user question
            conversation_history: List of conversation messages

        Returns:
            Reformulated standalone query
        """
        # Format conversation history
        history_text = ""
        for msg in conversation_history:
            speaker = msg.get('speaker', 'unknown')
            text = msg.get('text', '')
            history_text += f"{speaker}: {text}\n"

        prompt = f"""You are a query reformulation module for a hybrid RAG retrieval system (BM25 + Dense Vector Search).

Objective: Rewrite the current question into a self-contained search query that can retrieve relevant documents without needing conversation context.

Conversation History:
{history_text}

Current Question: {query}

Internal reasoning (do not output):
1. Identify pronouns, demonstratives, or implicit references (it, this, that, they, the one, etc.)
2. Resolve coreferences using conversation history
3. Replace references with explicit entity names, concepts, or descriptions
4. Maintain the original information need and question type
5. Optimize for keyword matching (BM25) and semantic search (vectors)
6. If question is already self-contained, keep it unchanged

Strict output rules:
- Output ONLY the rewritten query
- No explanations, no reasoning steps, no markdown, no labels
- One line only
- Must be a valid standalone search query
- Preserve domain-specific terminology
- Use declarative retrieval-style when possible (prefer statements over questions)
- Do NOT introduce information not implied by the conversation
- Do NOT change the semantic intent of the question

Examples:
Conversation: "The Arizona Cardinals played in London." | Question: "When was that game?"
→ Arizona Cardinals London game date

Conversation: "The Patriots won the Super Bowl." | Question: "Who was their quarterback?"
→ New England Patriots Super Bowl quarterback

Conversation: "FEMA recommends disaster supplies." | Question: "What should be in it?"
→ FEMA disaster supplies kit contents

Rewritten Query:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.4,  # Low temperature for consistency
                    "max_output_tokens": 4096
                }
            )

            # Extract and clean the rewritten query
            full_response = response.text.strip()

            # Remove common artifacts
            result = full_response.replace('*', '').replace('#', '').replace('Rewritten Query:', '')

            # Take only first line to avoid explanations
            result = result.split('\n')[0].strip()

            # Clean up quotes if present
            result = result.strip('"').strip("'")

            return result if result else query

        except Exception as e:
            print(f"Error in query reformulation: {e}")
            return query  # Return original query if reformulation fails


class QueryDiversifier:
    """Handles multi-strategy query diversification"""
    
    def __init__(self, config: QueryDiversificationConfig, text_model_config: TextModelConfig = None):
        """
        Initialize the query diversifier
        
        Args:
            config: Configuration for query diversification
            text_model_config: Configuration for text model
        """
        self.config = config
        self.text_model_config = text_model_config or TextModelConfig(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.client = genai.Client(api_key=self.text_model_config.api_key)
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def entity_focus(self, query: str) -> str:
        """Generate entity-focused query variant optimized for hybrid retrieval"""
        prompt = f"""You are a query generation module for a hybrid RAG retrieval system (BM25 + Dense Vector Search).

Objective: Generate ONE entity-focused search query optimized for retrieving relevant documents.

Internal reasoning (do not output):
- Identify explicit entities: people, organizations, products, locations, teams, technical terms
- Extract proper nouns and named entities
- Optimize for BM25 (keyword matching) and vector search (semantic similarity)

Strict output rules:
- Output ONLY the generated query
- No explanations, no markdown, no labels
- One line only
- Must be a valid standalone search query
- Prefer noun phrases and entity names
- Avoid conversational language
- Use declarative retrieval-style (not questions)

Input question: {query}

Entity-focused query:"""

        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 4096
                }
            )
            result = response.text.strip()
            # Clean up any extra formatting
            result = result.replace('*', '').replace('#', '').split('\n')[0].strip()
            return result if result else query
        except Exception as e:
            print(f"Error in entity focus generation: {e}")
            return query
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def action_focus(self, query: str) -> str:
        """Generate action-focused query variant optimized for hybrid retrieval"""
        prompt = f"""You are a query generation module for a hybrid RAG retrieval system (BM25 + Dense Vector Search).

Objective: Generate ONE action-focused search query optimized for retrieving relevant documents.

Internal reasoning (do not output):
- Identify main actions, events, processes, procedures, or operations
- Extract verbs indicating activities (play, move, win, create, implement, etc.)
- Focus on "what happened" or "how to do something"
- Optimize for both keyword matching (BM25) and semantic search (vectors)

Strict output rules:
- Output ONLY the generated query
- No explanations, no markdown, no labels
- One line only
- Must be a valid standalone search query
- Emphasize action verbs and process descriptions
- Avoid conversational language
- Use declarative retrieval-style (not questions)

Input question: {query}

Action-focused query:"""

        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 4096
                }
            )
            result = response.text.strip()
            result = result.replace('*', '').replace('#', '').split('\n')[0].strip()
            return result if result else query
        except Exception as e:
            print(f"Error in action focus generation: {e}")
            return query
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def paraphrase(self, query: str) -> str:
        """Generate paraphrased query variant optimized for hybrid retrieval"""
        prompt = f"""You are a query generation module for a hybrid RAG retrieval system (BM25 + Dense Vector Search).

Objective: Generate ONE paraphrased search query optimized for retrieving relevant documents.

Internal reasoning (do not output):
- Rephrase using different words while maintaining semantic meaning
- Use synonyms and alternative expressions
- Vary sentence structure but keep information need constant
- Optimize for semantic similarity (dense vectors) while preserving key terms (BM25)
- Do NOT change entity names or proper nouns

Strict output rules:
- Output ONLY the generated query
- No explanations, no markdown, no labels
- One line only
- Must be a valid standalone search query
- Keep critical domain-specific terms unchanged
- Avoid conversational language
- Use declarative retrieval-style (not questions)

Input question: {query}

Paraphrased query:"""

        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.4,  # Slightly higher for lexical diversity
                    "max_output_tokens": 4096
                }
            )
            result = response.text.strip()
            result = result.replace('*', '').replace('#', '').split('\n')[0].strip()
            return result if result else query
        except Exception as e:
            print(f"Error in paraphrase generation: {e}")
            return query
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def relation_focus(self, query: str) -> str:
        """Generate relation-focused query variant optimized for hybrid retrieval"""
        prompt = f"""You are a query generation module for a hybrid RAG retrieval system (BM25 + Dense Vector Search).

Objective: Generate ONE relation-focused search query optimized for retrieving relevant documents.

Internal reasoning (do not output):
- Identify relationships, connections, dependencies, or comparisons between entities/concepts
- Extract semantic relations: causes, enables, depends on, compares, belongs to, results in
- Focus on "how X relates to Y" or "connection between X and Y"
- Optimize for both keyword matching (BM25) and semantic relation understanding (vectors)

Strict output rules:
- Output ONLY the generated query
- No explanations, no markdown, no labels
- One line only
- Must be a valid standalone search query
- Emphasize relational verbs and connective phrases
- Avoid conversational language
- Use declarative retrieval-style (not questions)

Input question: {query}

Relation-focused query:"""

        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 4096
                }
            )
            result = response.text.strip()
            result = result.replace('*', '').replace('#', '').split('\n')[0].strip()
            return result if result else query
        except Exception as e:
            print(f"Error in relation focus generation: {e}")
            return query
    
    def diversify(self, query: str) -> List[str]:
        """
        Generate diverse query variants
        
        Args:
            query: Reformulated query
            
        Returns:
            List of query variants including the original
        """
        variants = [query]  # Always include the original reformulated query
        
        if self.config.use_entity_focus:
            variants.append(self.entity_focus(query))
        
        if self.config.use_action_focus:
            variants.append(self.action_focus(query))
        
        if self.config.use_paraphrase:
            variants.append(self.paraphrase(query))
        
        if self.config.use_relation_focus:
            variants.append(self.relation_focus(query))
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant and variant not in seen:
                unique_variants.append(variant)
                seen.add(variant)
        
        # Limit to configured number of variants
        return unique_variants[:self.config.num_variants]
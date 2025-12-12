"""
Answerability Detection Module

This module determines whether a query can be answered given the retrieved documents.
"""

import os
import logging
from typing import List, Dict, Any, Literal
from enum import Enum
from dotenv import load_dotenv
from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)
from .config import TextModelConfig

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
    ])


class AnswerabilityType(Enum):
    """Enumeration for answerability types"""
    ANSWERABLE = "A"
    PARTIAL = "P"
    UNANSWERABLE = "U"


class AnswerabilityDetector:
    """Detects whether a query can be answered given retrieved documents"""
    
    def __init__(self, config: TextModelConfig = None):
        """Initialize the answerability detector"""
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
    def classify(self, query: str, documents: List[Dict[str, Any]]) -> AnswerabilityType:
        """
        Classify whether the query can be answered using the retrieved documents

        Args:
            query: The reformulated query
            documents: List of retrieved documents

        Returns:
            AnswerabilityType indicating if query is answerable, partially answerable, or unanswerable
        """
        if not documents:
            return AnswerabilityType.UNANSWERABLE

        # Format documents for the prompt
        docs_text = ""
        for i, doc in enumerate(documents[:5]):  # Limit to top 5 for context window
            docs_text += f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nContent: {doc['text'][:500]}...\n\n"

        prompt = f"""Given the following query and retrieved documents, classify whether the query can be answered.

Query: {query}

Documents:
{docs_text}

Classification options:
- Answerable: Documents contain sufficient information to provide a complete answer
- Partial: Documents contain some relevant information but missing key details
- Unanswerable: Documents do not contain relevant information to answer the query

Provide your classification and brief reasoning:

Classification:
Reasoning:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.0,  # Use 0 temperature for classification
                    "max_output_tokens": 512
                }
            )

            response_text = response.text.strip().lower()

            # Parse the classification from the response
            if "answerable" in response_text:
                if "partial" in response_text or "partially" in response_text:
                    return AnswerabilityType.PARTIAL
                elif "unanswerable" in response_text or "not" in response_text:
                    return AnswerabilityType.UNANSWERABLE
                else:
                    return AnswerabilityType.ANSWERABLE
            elif "partial" in response_text:
                return AnswerabilityType.PARTIAL
            elif "unanswerable" in response_text:
                return AnswerabilityType.UNANSWERABLE
            else:
                # Default to answerable if we can't parse the response
                return AnswerabilityType.ANSWERABLE

        except Exception as e:
            print(f"Error in answerability classification: {e}")
            # Default to answerable on error
            return AnswerabilityType.ANSWERABLE
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def get_classification_confidence(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed classification with confidence and reasoning

        Args:
            query: The reformulated query
            documents: List of retrieved documents

        Returns:
            Dictionary with classification, confidence, and reasoning
        """
        if not documents:
            return {
                "classification": AnswerabilityType.UNANSWERABLE,
                "confidence": 1.0,
                "reasoning": "No documents retrieved"
            }

        # Format documents for the prompt
        docs_text = ""
        for i, doc in enumerate(documents[:5]):
            docs_text += f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nContent: {doc['text'][:500]}...\n\n"

        prompt = f"""Given the following query and retrieved documents, classify whether the query can be answered and provide a confidence score.

Query: {query}

Documents:
{docs_text}

Please provide:
1. Classification (Answerable/Partial/Unanswerable)
2. Confidence score (0.0 to 1.0)
3. Brief reasoning

Format your response as:
Classification: [your classification]
Confidence: [0.0-1.0]
Reasoning: [your reasoning]"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.0,  # Use 0 temperature for classification
                    "max_output_tokens": 512
                }
            )

            response_text = response.text.strip()

            # Parse the response
            classification = AnswerabilityType.ANSWERABLE  # default
            confidence = 0.8  # default
            reasoning = "Could not parse response"

            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('Classification:'):
                    class_text = line.split(':', 1)[1].strip().lower()
                    if "partial" in class_text:
                        classification = AnswerabilityType.PARTIAL
                    elif "unanswerable" in class_text:
                        classification = AnswerabilityType.UNANSWERABLE
                    else:
                        classification = AnswerabilityType.ANSWERABLE
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    except:
                        confidence = 0.8
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()

            return {
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            print(f"Error in detailed answerability classification: {e}")
            return {
                "classification": AnswerabilityType.ANSWERABLE,
                "confidence": 0.5,
                "reasoning": f"Error in classification: {e}"
            }
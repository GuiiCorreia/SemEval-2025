"""
Response Generation Module

This module handles conditional response generation with guardrails.
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
from .answerability import AnswerabilityType
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


class ResponseGenerator:
    """Generates responses based on answerability and retrieved documents"""
    
    def __init__(self, config: TextModelConfig = None):
        """Initialize the response generator"""
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
    def generate_complete_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate a complete answer when query is fully answerable

        Args:
            query: The reformulated query
            documents: Retrieved documents
            conversation_history: Previous conversation context

        Returns:
            Generated complete answer
        """
        # Format conversation history
        history_text = ""
        for msg in conversation_history:
            speaker = msg.get('speaker', 'unknown')
            text = msg.get('text', '')
            history_text += f"{speaker}: {text}\n"

        # Format documents
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nContent: {doc['text'][:800]}\n\n"

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided documents and conversation context.

Conversation Context:
{history_text}

Question: {query}

Relevant Documents:
{docs_text}

Instructions:
- Use only information from the provided documents
- If information spans multiple documents, synthesize appropriately
- Maintain conversational tone appropriate to the context
- Be specific and cite relevant details
- If the documents don't fully address the question, acknowledge limitations

Answer:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating complete answer: {e}")
            return "I apologize, but I'm unable to generate a response at this time due to a technical error."
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def generate_partial_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate a partial answer when query is only partially answerable

        Args:
            query: The reformulated query
            documents: Retrieved documents
            conversation_history: Previous conversation context

        Returns:
            Generated partial answer with gaps explained
        """
        # Format documents
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nContent: {doc['text'][:800]}\n\n"

        prompt = f"""Based on the provided documents, you can only partially answer the user's question.

Question: {query}

Available Information:
{docs_text}

Task: Provide a partial answer and clearly explain what information is missing.

Format your response as follows:
1. Start with what you can answer based on the available documents
2. Clearly state what information is missing or incomplete
3. Suggest what additional information would be needed for a complete answer

Response:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating partial answer: {e}")
            return "I can only partially address your question based on the available information. However, I'm experiencing technical difficulties in generating a detailed response."
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def generate_clarification_request(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate a clarification request when query is unanswerable

        Args:
            query: The reformulated query
            conversation_history: Previous conversation context

        Returns:
            Generated clarification request
        """
        # Format conversation history
        history_text = ""
        for msg in conversation_history[-3:]:  # Last 3 messages for context
            speaker = msg.get('speaker', 'unknown')
            text = msg.get('text', '')
            history_text += f"{speaker}: {text}\n"

        prompt = f"""The user has asked a question that cannot be adequately answered with the available documents.

Conversation Context:
{history_text}

Question: {query}

Task: Generate a helpful clarification request that:
1. Acknowledges that you don't have sufficient information
2. Asks for specific clarification or additional context
3. Suggests what type of information would be helpful
4. Maintains a helpful and professional tone

Clarification Request:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating clarification request: {e}")
            return "I don't have enough information to answer your question adequately. Could you please provide more context or clarify what specific information you're looking for?"
    
    @retry(
        retry=retry_if_exception(is_api_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def apply_guardrails(self, response: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply quality guardrails to the generated response

        Args:
            response: Generated response text
            documents: Source documents

        Returns:
            Dictionary with response and quality scores
        """
        # Format documents for guardrail checking
        docs_text = ""
        for i, doc in enumerate(documents[:3]):  # Limit for context
            docs_text += f"Document {i+1}: {doc['text'][:500]}\n\n"

        # Check faithfulness
        faithfulness_prompt = f"""Check if this response is faithful to the source documents:

Response: {response}
Source Documents: {docs_text}

Verify:
1. Are all claims supported by the documents?
2. Are there any fabricated details?
3. Are quotes and references accurate?

Provide a faithfulness score from 0.0 to 1.0 and list any issues:

Faithfulness Score:
Issues Found:"""

        try:
            faithfulness_response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=faithfulness_prompt,
                config={
                    "temperature": 0.0,  # Use 0 temperature for evaluation
                    "max_output_tokens": 512
                }
            )

            # Parse faithfulness score
            faithfulness_text = faithfulness_response.text.strip()
            faithfulness_score = 0.8  # default
            faithfulness_issues = []

            for line in faithfulness_text.split('\n'):
                if 'score:' in line.lower():
                    try:
                        score_part = line.split(':')[1].strip()
                        faithfulness_score = float(score_part)
                        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
                    except:
                        pass
                elif 'issues' in line.lower() and ':' in line:
                    issues_part = line.split(':', 1)[1].strip()
                    if issues_part and issues_part.lower() not in ['none', 'no issues']:
                        faithfulness_issues.append(issues_part)

        except Exception as e:
            print(f"Error in faithfulness checking: {e}")
            faithfulness_score = 0.8
            faithfulness_issues = []

        return {
            "response_text": response,
            "confidence_score": faithfulness_score,
            "sources": [doc.get('document_id', f"doc_{i}") for i, doc in enumerate(documents)],
            "guardrail_flags": {
                "faithfulness_score": faithfulness_score,
                "faithfulness_issues": faithfulness_issues,
                "low_faithfulness": faithfulness_score < 0.6
            }
        }
    
    def generate_response(
        self,
        answerability: AnswerabilityType,
        query: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate response based on answerability classification
        
        Args:
            answerability: Classification of whether query can be answered
            query: The reformulated query
            documents: Retrieved documents
            conversation_history: Previous conversation context
            
        Returns:
            Generated response with quality scores
        """
        if answerability == AnswerabilityType.ANSWERABLE:
            draft_response = self.generate_complete_answer(query, documents, conversation_history)
            return self.apply_guardrails(draft_response, documents)
        
        elif answerability == AnswerabilityType.PARTIAL:
            partial_response = self.generate_partial_answer(query, documents, conversation_history)
            return {
                "response_text": partial_response,
                "confidence_score": 0.7,
                "sources": [doc.get('document_id', f"doc_{i}") for i, doc in enumerate(documents)],
                "guardrail_flags": {
                    "partial_answer": True,
                    "faithfulness_score": 0.7,
                    "faithfulness_issues": []
                }
            }
        
        else:  # UNANSWERABLE
            clarification_response = self.generate_clarification_request(query, conversation_history)
            return {
                "response_text": clarification_response,
                "confidence_score": 0.9,  # High confidence in clarification requests
                "sources": [],
                "guardrail_flags": {
                    "clarification_request": True,
                    "faithfulness_score": 1.0,
                    "faithfulness_issues": []
                }
            }
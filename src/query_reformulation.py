"""
Query Reformulation Module

This module handles context-aware query reformulation and query diversification.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from .config import QueryDiversificationConfig, TextModelConfig

# Load environment variables
load_dotenv()


class QueryReformulator:
    """Handles query reformulation using Chain-of-Thought reasoning"""
    
    def __init__(self, config: TextModelConfig = None):
        """Initialize the query reformulator"""
        self.config = config or TextModelConfig(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.client = genai.Client(api_key=self.config.api_key)
    
    def cot_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Rewrite query using Chain-of-Thought reasoning
        
        Args:
            query: Current user question
            conversation_history: List of conversation messages
            
        Returns:
            Reformulated query
        """
        # Format conversation history
        history_text = ""
        for msg in conversation_history[-10:]:  # Limit to last 10 messages for context
            speaker = msg.get('speaker', 'unknown')
            text = msg.get('text', '')
            history_text += f"{speaker}: {text}\n"
        
        prompt = f"""You are an expert at understanding conversational context and rewriting queries.

Conversation History:
{history_text}

Current Question: {query}

Task: Rewrite the current question to be self-contained by:
1. Identifying any pronouns or references that need context
2. Finding what they refer to in the conversation history  
3. Replacing them with explicit mentions
4. Ensuring the rewritten query can be understood without the conversation context

Think step by step:
Step 1: What pronouns/references need clarification?
Step 2: What do they refer to based on the conversation?
Step 3: How should I rewrite this to be self-contained?

Rewritten Query:"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.2,  # Slightly higher for creativity in rewriting
                    "max_output_tokens": 512
                }
            )
            
            # Extract just the rewritten query from the response
            full_response = response.text.strip()
            if "Rewritten Query:" in full_response:
                rewritten = full_response.split("Rewritten Query:")[-1].strip()
            else:
                # If the model doesn't follow the format, return the full response
                rewritten = full_response
            
            return rewritten if rewritten else query
            
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
    
    def entity_focus(self, query: str) -> str:
        """Generate entity-focused query variant"""
        prompt = f"""Generate an entity-focused variation of this query: {query}

Focus on the main entities (people, organizations, products, locations) mentioned.
Make the query more specific about these entities.

Entity-focused query:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.3,  # Higher temperature for diversity
                    "max_output_tokens": 256
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in entity focus generation: {e}")
            return query
    
    def action_focus(self, query: str) -> str:
        """Generate action-focused query variant"""
        prompt = f"""Generate an action-focused variation of this query: {query}

Focus on the main actions, processes, or procedures mentioned.
Make the query more specific about what actions are being asked about.

Action-focused query:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.3,  # Higher temperature for diversity
                    "max_output_tokens": 256
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in action focus generation: {e}")
            return query
    
    def paraphrase(self, query: str) -> str:
        """Generate paraphrased query variant"""
        prompt = f"""Paraphrase this query while keeping the same meaning: {query}

Use different words and sentence structure but maintain the original intent.

Paraphrased query:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.3,  # Higher temperature for diversity
                    "max_output_tokens": 256
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in paraphrase generation: {e}")
            return query
    
    def relation_focus(self, query: str) -> str:
        """Generate relation-focused query variant"""
        prompt = f"""Generate a relationship-focused variation of this query: {query}

Focus on relationships, connections, or dependencies between concepts mentioned.
Make the query more specific about how things relate to each other.

Relation-focused query:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.text_model_config.model_id,
                contents=prompt,
                config={
                    "temperature": 0.3,  # Higher temperature for diversity
                    "max_output_tokens": 256
                }
            )
            return response.text.strip()
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
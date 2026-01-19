"""
DSPy Classification and Response Generation Modules

This module implements:
1. AnswerabilityClassifier - Classifies questions as ANSWERABLE, UNANSWERABLE, CONVERSATIONAL, or PARTIAL
2. QuestionTypeClassifier - Classifies question types (multi-label)
3. UnifiedClassifier - Combined classifier for both answerability and question type
4. ResponseGenerator - Generates responses based on classification and context
"""

import dspy
from typing import Literal, List, Dict, Any


# =============================================================================
# QUESTION TYPE DEFINITIONS
# =============================================================================

QUESTION_TYPES = [
    "Comparative",    # Asking for comparison of entities/concepts/characteristics
    "Composite",      # Comprises several questions, may be related or dependent
    "Explanation",    # Explain the reason behind something
    "Factoid",        # Asking for specific piece of information (date, quantity, name, yes/no)
    "How-To",         # Instructions describing how to perform a task
    "Keyword",        # Asking using keywords, not full sentence/phrase
    "Non-question",   # Not asking a question, providing information
    "Opinion",        # Asking model's opinion on something
    "Summarization",  # Asking to summarize a process or policy
    "Troubleshooting" # Finding solutions to issues, problems, challenges
]


# =============================================================================
# ANSWERABILITY CLASSIFICATION
# =============================================================================

class ClassifyAnswerability(dspy.Signature):
    """Classify the answerability of a question based on conversation history and retrieved documents.

    Answerability Types:
    - ANSWERABLE: The question can be fully answered from the provided passages/documents.
    - PARTIAL: Only part of the question can be answered from the passages. Some information is available but not complete.
    - UNANSWERABLE: The question cannot be answered neither fully nor partially from the passages.
    - CONVERSATIONAL: The user turn does not contain a question but is a conversational statement (e.g., "Hello", "Hi, I had a question", "Cool", "That's interesting", "That was all", "Thank you").
    """

    conversation_history: str = dspy.InputField(desc="Previous conversation turns for context")
    current_question: str = dspy.InputField(desc="The user's current question or statement")
    retrieved_documents: str = dspy.InputField(desc="Retrieved documents/passages that may contain relevant information")

    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning about the answerability based on the available passages")
    classification: Literal["ANSWERABLE", "UNANSWERABLE", "CONVERSATIONAL", "PARTIAL"] = dspy.OutputField(
        desc="The answerability classification"
    )


class AnswerabilityClassifier(dspy.Module):
    """Classifies questions based on answerability given context."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyAnswerability)

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        retrieved_documents: str
    ) -> dspy.Prediction:
        result = self.classify(
            conversation_history=conversation_history,
            current_question=current_question,
            retrieved_documents=retrieved_documents
        )

        return dspy.Prediction(
            classification=result.classification,
            reasoning=result.reasoning
        )


# =============================================================================
# QUESTION TYPE CLASSIFICATION (MULTI-LABEL)
# =============================================================================

class ClassifyQuestionType(dspy.Signature):
    """Classify the type(s) of a question. A question can have multiple types (multi-label classification).

    Question Types:
    - Comparative: Asking for comparison. Can be (a) comparison of multiple entities/concepts, (b) comparison of characteristics of a single entity, or (c) comparison with decision (e.g., "is X better than Y").
    - Composite: Comprises several questions that may be related or dependent (e.g., "Am I eligible for a driver's license and how do I apply?").
    - Explanation: Asking to explain the reason behind something (e.g., "Why do I have to do X?").
    - Factoid: Asking for a specific piece of information such as a date, quantity, name, yes/no answer, or other singular fact. Can be answered directly and concisely without requiring explanation or opinion.
    - How-To: Instructions describing how to perform a task (e.g., "How do I apply for benefits?").
    - Keyword: Asking using keywords only, not a full sentence/phrase. May be ambiguous (e.g., "vacation days", "ios 17 upgrade").
    - Non-question: Not asking a question but instead answering or providing information asked by the model.
    - Opinion: Asking for the model's opinion on something. May be phrased as leading (e.g., "Don't you think X is better?").
    - Summarization: Asking to summarize a process or policy (e.g., "What's the policy on vacation days?").
    - Troubleshooting: Finding solutions to issues, problems, or challenges (e.g., "I have error X, what should I do?").

    Output the types as a comma-separated list. A question can have multiple types.
    """

    conversation_history: str = dspy.InputField(desc="Previous conversation turns for context")
    current_question: str = dspy.InputField(desc="The user's current question or statement")

    reasoning: str = dspy.OutputField(desc="Step-by-step analysis of what type(s) of question this is")
    question_types: str = dspy.OutputField(desc="Comma-separated list of applicable question types (e.g., 'Factoid, Keyword' or 'How-To')")


class QuestionTypeClassifier(dspy.Module):
    """Classifies question types (multi-label)."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyQuestionType)

    def forward(
        self,
        conversation_history: str,
        current_question: str
    ) -> dspy.Prediction:
        result = self.classify(
            conversation_history=conversation_history,
            current_question=current_question
        )

        # Parse comma-separated types into a list
        types_str = result.question_types
        question_types = [t.strip() for t in types_str.split(',') if t.strip()]

        # Validate types against known types
        valid_types = []
        for t in question_types:
            # Find best match (case-insensitive)
            for known_type in QUESTION_TYPES:
                if t.lower() == known_type.lower():
                    valid_types.append(known_type)
                    break

        return dspy.Prediction(
            question_types=valid_types,
            question_types_str=', '.join(valid_types),
            reasoning=result.reasoning
        )


# =============================================================================
# UNIFIED CLASSIFIER (ANSWERABILITY + QUESTION TYPE)
# =============================================================================

class ClassifyUnified(dspy.Signature):
    """Classify both the answerability and question type(s) of a user query.

    ANSWERABILITY - Based on whether the question can be answered from the provided passages:
    - ANSWERABLE: The question can be fully answered from the provided passages/documents.
    - PARTIAL: Only part of the question can be answered from the passages.
    - UNANSWERABLE: The question cannot be answered from the passages.
    - CONVERSATIONAL: The user turn is a conversational statement, not a question (e.g., "Hello", "Thank you").

    QUESTION TYPES (can have multiple, comma-separated):
    - Comparative: Asking for comparison of entities, characteristics, or decisions.
    - Composite: Multiple questions in one, may be related or dependent.
    - Explanation: Asking to explain the reason behind something.
    - Factoid: Asking for specific information (date, quantity, name, yes/no).
    - How-To: Instructions on how to perform a task.
    - Keyword: Using keywords only, not full sentences.
    - Non-question: Providing information, not asking a question.
    - Opinion: Asking for opinion or preference.
    - Summarization: Asking to summarize a process or policy.
    - Troubleshooting: Finding solutions to problems or issues.
    """

    conversation_history: str = dspy.InputField(desc="Previous conversation turns for context")
    current_question: str = dspy.InputField(desc="The user's current question or statement")
    retrieved_documents: str = dspy.InputField(desc="Retrieved documents/passages for answering")

    answerability_reasoning: str = dspy.OutputField(desc="Reasoning about answerability based on passages")
    answerability: Literal["ANSWERABLE", "UNANSWERABLE", "CONVERSATIONAL", "PARTIAL"] = dspy.OutputField(
        desc="The answerability classification"
    )
    question_type_reasoning: str = dspy.OutputField(desc="Reasoning about the question type(s)")
    question_types: str = dspy.OutputField(desc="Comma-separated list of question types")


class UnifiedClassifier(dspy.Module):
    """Combined classifier for answerability and question type."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyUnified)

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        retrieved_documents: str
    ) -> dspy.Prediction:
        result = self.classify(
            conversation_history=conversation_history,
            current_question=current_question,
            retrieved_documents=retrieved_documents
        )

        # Parse question types
        types_str = result.question_types
        question_types = [t.strip() for t in types_str.split(',') if t.strip()]

        # Validate types
        valid_types = []
        for t in question_types:
            for known_type in QUESTION_TYPES:
                if t.lower() == known_type.lower():
                    valid_types.append(known_type)
                    break

        return dspy.Prediction(
            answerability=result.answerability,
            answerability_reasoning=result.answerability_reasoning,
            question_types=valid_types,
            question_types_str=', '.join(valid_types),
            question_type_reasoning=result.question_type_reasoning
        )


# =============================================================================
# RESPONSE GENERATION MODULE
# =============================================================================

class GenerateResponse(dspy.Signature):
    """Generate a response to the user's question based on classification, question type, and context.

    Response strategy based on ANSWERABILITY:
    - ANSWERABLE: Provide a comprehensive answer using the context.
    - UNANSWERABLE: Politely explain that the information is not available in the provided context.
    - CONVERSATIONAL: Respond appropriately to the conversational statement (greetings, thanks, etc.).
    - PARTIAL: Provide what information is available and clearly note what information is missing.

    Response style based on QUESTION TYPE:
    - Comparative: Structure the response to highlight differences/similarities.
    - Composite: Address each sub-question systematically.
    - Explanation: Provide clear reasoning and explanations.
    - Factoid: Give a direct, concise answer.
    - How-To: Provide step-by-step instructions.
    - Keyword: Interpret the keywords and provide relevant information.
    - Non-question: Acknowledge the information provided by the user.
    - Opinion: Provide objective information rather than opinions.
    - Summarization: Provide a structured summary.
    - Troubleshooting: Provide diagnostic steps and solutions.
    """

    conversation_history: str = dspy.InputField(desc="Previous conversation turns for context")
    current_question: str = dspy.InputField(desc="The user's current question or statement")
    retrieved_documents: str = dspy.InputField(desc="Retrieved documents containing relevant information")
    answerability: str = dspy.InputField(desc="The answerability classification (ANSWERABLE, UNANSWERABLE, CONVERSATIONAL, PARTIAL)")
    question_types: str = dspy.InputField(desc="The question type(s), comma-separated (e.g., 'Factoid', 'How-To, Composite')")

    response: str = dspy.OutputField(desc="The generated response to the user, tailored to the answerability and question type")


class ResponseGenerator(dspy.Module):
    """Generates responses based on classification, question type, and context."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateResponse)

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        retrieved_documents: str,
        answerability: str,
        question_types: str = ""
    ) -> dspy.Prediction:
        """
        Generate a response based on classifications.

        Args:
            conversation_history: Formatted conversation history
            current_question: The user's current question
            retrieved_documents: Formatted retrieved documents
            answerability: The answerability classification
            question_types: Comma-separated question types

        Returns:
            Prediction with generated response
        """
        result = self.generate(
            conversation_history=conversation_history,
            current_question=current_question,
            retrieved_documents=retrieved_documents,
            answerability=answerability,
            question_types=question_types
        )

        return dspy.Prediction(response=result.response)


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

class ClassifyAndRespond(dspy.Module):
    """
    Combined pipeline that classifies (answerability + question type) and generates response.
    Uses different LMs for classification (fast) and generation (powerful).
    """

    def __init__(
        self,
        classification_lm: dspy.LM = None,
        generation_lm: dspy.LM = None,
        use_unified_classifier: bool = True
    ):
        """
        Initialize the combined pipeline.

        Args:
            classification_lm: LM for classification (default: gemini-3-flash-preview)
            generation_lm: LM for generation (default: gemini-3-pro-preview)
            use_unified_classifier: If True, use UnifiedClassifier for both answerability and question type
        """
        super().__init__()
        self.use_unified = use_unified_classifier
        if use_unified_classifier:
            self.classifier = UnifiedClassifier()
        else:
            self.classifier = AnswerabilityClassifier()
        self.generator = ResponseGenerator()
        self.classification_lm = classification_lm
        self.generation_lm = generation_lm

    def forward(
        self,
        conversation_history: str,
        current_question: str,
        retrieved_documents: str
    ) -> dspy.Prediction:
        """
        Classify and generate response.

        Args:
            conversation_history: Formatted conversation history
            current_question: The user's current question
            retrieved_documents: Formatted retrieved documents

        Returns:
            Prediction with answerability, question_types, reasoning, and response
        """
        # First classify (using classification LM if provided)
        if self.classification_lm:
            with dspy.context(lm=self.classification_lm):
                classification_result = self.classifier(
                    conversation_history=conversation_history,
                    current_question=current_question,
                    retrieved_documents=retrieved_documents
                )
        else:
            classification_result = self.classifier(
                conversation_history=conversation_history,
                current_question=current_question,
                retrieved_documents=retrieved_documents
            )

        # Extract classification results
        if self.use_unified:
            answerability = classification_result.answerability
            question_types_str = classification_result.question_types_str
            question_types = classification_result.question_types
        else:
            answerability = classification_result.classification
            question_types_str = ""
            question_types = []

        # Then generate response based on classification (using generation LM if provided)
        if self.generation_lm:
            with dspy.context(lm=self.generation_lm):
                response_result = self.generator(
                    conversation_history=conversation_history,
                    current_question=current_question,
                    retrieved_documents=retrieved_documents,
                    answerability=answerability,
                    question_types=question_types_str
                )
        else:
            response_result = self.generator(
                conversation_history=conversation_history,
                current_question=current_question,
                retrieved_documents=retrieved_documents,
                answerability=answerability,
                question_types=question_types_str
            )

        return dspy.Prediction(
            answerability=answerability,
            question_types=question_types,
            question_types_str=question_types_str,
            response=response_result.response
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_conversation_history(conversation: List[Dict[str, str]]) -> str:
    """Format conversation history as a string."""
    if not conversation:
        return "No previous conversation."

    formatted = []
    for msg in conversation:
        speaker = msg.get('speaker', 'unknown')
        text = msg.get('text', '')
        formatted.append(f"{speaker}: {text}")

    return "\n".join(formatted)


def format_documents(documents: List[Dict[str, Any]]) -> str:
    """Format documents for the prompt."""
    if not documents:
        return "No documents provided."

    formatted = []
    for i, doc in enumerate(documents, 1):
        title = doc.get('title', 'Untitled')
        text = doc.get('text', '')  # Truncate long texts
        formatted.append(f"[Document {i}] {title}\n{text}")

    return "\n\n".join(formatted)


def create_classifier(unified: bool = True):
    """Factory function to create a classifier.

    Args:
        unified: If True, returns UnifiedClassifier (both answerability and question type).
                 If False, returns AnswerabilityClassifier only.
    """
    if unified:
        return UnifiedClassifier()
    return AnswerabilityClassifier()


def create_question_type_classifier() -> QuestionTypeClassifier:
    """Factory function to create a question type classifier."""
    return QuestionTypeClassifier()


def create_generator() -> ResponseGenerator:
    """Factory function to create a response generator."""
    return ResponseGenerator()


def create_pipeline(
    classification_lm: dspy.LM = None,
    generation_lm: dspy.LM = None,
    api_key: str = None
) -> ClassifyAndRespond:
    """
    Factory function to create the combined pipeline with configured LMs.

    Args:
        classification_lm: LM for classification (default: creates gemini-3-flash-preview)
        generation_lm: LM for generation (default: creates gemini-3-pro-preview)
        api_key: API key for Gemini (required if LMs not provided)

    Returns:
        ClassifyAndRespond pipeline with configured LMs
    """
    if classification_lm is None and api_key:
        classification_lm = dspy.LM(
            "gemini/gemini-3-flash-preview",
            api_key=api_key,
            temperature=0.2,
            max_tokens=2000
        )

    if generation_lm is None and api_key:
        generation_lm = dspy.LM(
            "gemini/gemini-3-pro-preview",
            api_key=api_key,
            temperature=0.4,
            max_tokens=4000
        )

    return ClassifyAndRespond(
        classification_lm=classification_lm,
        generation_lm=generation_lm
    )
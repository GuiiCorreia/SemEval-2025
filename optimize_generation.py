"""
Response Generation Module Optimization Script

This script uses DSPy GEPA optimizer to improve the ResponseGenerator
using an LLM judge that compares generated responses with ground truth.

Usage:
    python optimize_generation.py --reference-file human/generation_tasks/RAG_clapnq.jsonl --samples 100
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

import dspy
from dspy import GEPA
from dotenv import load_dotenv

from src.dspy_classification import (
    ResponseGenerator,
    format_conversation_history,
    format_documents
)

# Load environment variables
load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

STUDENT_MODEL = "gemini/gemini-3-pro-preview"
TEACHER_MODEL = "gemini/gemini-3-pro-preview"
JUDGE_MODEL = "gemini/gemini-3-flash-preview"

STUDENT_TEMPERATURE = 0.4
TEACHER_TEMPERATURE = 0.4
JUDGE_TEMPERATURE = 0.2

MAX_TOKENS = 4000

# Create LM instances
lm_student = dspy.LM(STUDENT_MODEL, temperature=STUDENT_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=GEMINI_API_KEY, cache=False)
lm_teacher = dspy.LM(TEACHER_MODEL, temperature=TEACHER_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=GEMINI_API_KEY, cache=False)
lm_judge = dspy.LM(JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, max_tokens=2000, api_key=GEMINI_API_KEY, cache=False)

# Configure default LM
dspy.configure(lm=lm_student)


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class GenerationSample:
    """A training sample for generation optimization."""
    task_id: str
    collection: str
    question: str
    conversation_history: List[Dict[str, str]]
    documents: List[Dict[str, Any]]
    classification: str  # Ground truth classification
    ground_truth_response: str  # Expected response from targets


def load_reference_data(filepath: str) -> List[Dict[str, Any]]:
    """Load reference data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # Only include entries that have targets with agent response
            targets = entry.get('targets', [])
            if targets and targets[0].get('speaker') == 'agent' and targets[0].get('text'):
                data.append(entry)
    return data


def stratified_sample(
    data: List[Dict[str, Any]],
    n_samples: int = 100,
    seed: int = 42
) -> List[GenerationSample]:
    """
    Create stratified sample balanced by Answerability label.

    Args:
        data: Full dataset
        n_samples: Total number of samples to select
        seed: Random seed for reproducibility

    Returns:
        List of GenerationSample objects
    """
    random.seed(seed)

    # Group by Answerability label
    groups = defaultdict(list)
    for entry in data:
        label = entry.get('Answerability', ['UNKNOWN'])[0]
        groups[label].append(entry)

    # Calculate samples per group
    n_groups = len(groups)
    base_per_group = n_samples // n_groups
    remainder = n_samples % n_groups

    print(f"Stratified sampling: {n_samples} samples across {n_groups} labels")
    for label, group_data in groups.items():
        print(f"  {label}: {len(group_data)} available")

    # Sample from each group
    samples = []
    labels = sorted(groups.keys())

    for i, label in enumerate(labels):
        group_data = groups[label]
        n_for_group = base_per_group + (1 if i < remainder else 0)
        n_for_group = min(n_for_group, len(group_data))

        selected = random.sample(group_data, n_for_group)

        for entry in selected:
            # Extract question from last user input
            inputs = entry.get('input', [])
            question = inputs[-1].get('text', '') if inputs else ''

            # Build conversation history (all but last message)
            conversation_history = inputs[:-1] if len(inputs) > 1 else []

            # Get documents (ground truth contexts)
            documents = entry.get('contexts', [])

            # Get ground truth response
            targets = entry.get('targets', [])
            ground_truth_response = ''
            for target in targets:
                if target.get('speaker') == 'agent':
                    ground_truth_response = target.get('text', '')
                    break

            sample = GenerationSample(
                task_id=entry.get('task_id', ''),
                collection=entry.get('Collection', ''),
                question=question,
                conversation_history=conversation_history,
                documents=documents,
                classification=label,
                ground_truth_response=ground_truth_response
            )
            samples.append(sample)

    print(f"Total samples selected: {len(samples)}")
    return samples


# =============================================================================
# LLM JUDGE FOR RESPONSE EVALUATION
# =============================================================================

class ResponseJudge(dspy.Signature):
    """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question given the provided document and a reference answer.

Your evaluation should assess the faithfulness, appropriateness, and completeness. Your evaluation should focus on the assistant's answer to the question of the current turn. You will be given the assistant's answer and a reference answer. You will also be given the user questions and assistant's answers of the previous turns of the conversation. You should consider how well the assistant's answer captures the key information, knowledge points mentioned in the reference answer, and how it respects or builds upon the focus and knowledge points from the previous turns.

[Faithfulness]: You are given the full conversation, the question of the current turn, the assistant's answer, and documents. You should evaluate how faithful is the assistant's answer to the information in the document and previous conversation.

[Appropriateness]: You should evaluate if the assistant's answer is relevant to the question of the current turn and if it addresses all the issues raised by the question without adding extra information.

[Completeness]: You should evaluate whether the assistant's answer is complete with information from the documents.

Begin your evaluation by comparing the assistant's answer against the reference answer in this turn. Be as objective as possible, and provide a detailed justification for your verdict."""

    previous_conversation: str = dspy.InputField(desc="Previous conversation turns")
    current_question: str = dspy.InputField(desc="The question of the current turn")
    reference_answer: str = dspy.InputField(desc="The reference/ground truth answer")
    assistant_answer: str = dspy.InputField(desc="The assistant's generated answer")
    documents: str = dspy.InputField(desc="The documents/passages used as context")

    evaluation: str = dspy.OutputField(desc="Detailed evaluation of faithfulness, appropriateness, and completeness")
    verdict: str = dspy.OutputField(desc="PASS if the response adequately captures the key information from the reference, FAIL otherwise")
    feedback: str = dspy.OutputField(desc="Specific suggestions for improving the response")


class LLMJudge(dspy.Module):
    """LLM-based judge for evaluating response quality."""

    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(ResponseJudge)

    def forward(
        self,
        previous_conversation: str,
        current_question: str,
        reference_answer: str,
        assistant_answer: str,
        documents: str
    ) -> dspy.Prediction:
        """
        Judge response quality.

        Args:
            previous_conversation: Previous conversation turns
            current_question: The current question
            reference_answer: Expected/ground truth response
            assistant_answer: System-generated response
            documents: Context documents

        Returns:
            Prediction with verdict and feedback
        """
        result = self.judge(
            previous_conversation=previous_conversation,
            current_question=current_question,
            reference_answer=reference_answer,
            assistant_answer=assistant_answer,
            documents=documents
        )

        # Parse verdict
        verdict_lower = result.verdict.lower()
        passed = 'pass' in verdict_lower and 'fail' not in verdict_lower

        return dspy.Prediction(
            evaluation=result.evaluation,
            verdict=result.verdict,
            feedback=result.feedback,
            passed=passed
        )


# =============================================================================
# METRIC WITH FEEDBACK FOR GEPA
# =============================================================================

def create_metric_with_feedback(judge: LLMJudge):
    """
    Create a metric function with feedback for GEPA optimization.

    Args:
        judge: LLMJudge instance

    Returns:
        Metric function compatible with GEPA (5 arguments)
    """
    def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """
        Evaluate response quality using LLM judge.

        Args:
            example: Training example with ground truth response
            prediction: Model prediction with generated response

        Returns:
            dspy.Prediction with score and feedback
        """
        # Get generated response
        generated = prediction.response if hasattr(prediction, 'response') else ''

        # Get ground truth from example
        reference_answer = example.ground_truth_response
        current_question = example.current_question
        previous_conversation = example.conversation_history
        documents = example.retrieved_documents

        if not generated:
            return dspy.Prediction(
                score=0,
                feedback="No response generated."
            )

        # Use LLM judge for evaluation
        with dspy.context(lm=lm_judge):
            judge_result = judge(
                previous_conversation=previous_conversation,
                current_question=current_question,
                reference_answer=reference_answer,
                assistant_answer=generated,
                documents=documents
            )

        score = 1.0 if judge_result.passed else 0.0
        feedback = f"{judge_result.evaluation}\n\nFeedback: {judge_result.feedback}"

        return dspy.Prediction(score=score, feedback=feedback)

    return metric_with_feedback


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def run_optimization(
    samples: List[GenerationSample],
    output_path: str = "optimized_generator.json",
    num_threads: int = 4
):
    """
    Run GEPA optimization on the generator.

    Args:
        samples: Training samples
        output_path: Path to save optimized model
        num_threads: Number of parallel threads
    """
    print("\n" + "="*60)
    print("STARTING GENERATION OPTIMIZATION")
    print("="*60)

    # Create the generator module
    generator = ResponseGenerator()

    # Create judge and metric
    judge = LLMJudge()
    metric = create_metric_with_feedback(judge)

    # Convert samples to dspy.Example format
    trainset = []
    for sample in samples:
        # Format inputs
        conv_history = format_conversation_history(sample.conversation_history)
        docs = format_documents(sample.documents)

        ex = dspy.Example(
            conversation_history=conv_history,
            current_question=sample.question,
            retrieved_documents=docs,
            classification=sample.classification,
            ground_truth_response=sample.ground_truth_response,
            task_id=sample.task_id
        ).with_inputs('conversation_history', 'current_question', 'retrieved_documents', 'classification')
        trainset.append(ex)

    # Split into train and validation
    random.shuffle(trainset)
    split_idx = int(len(trainset) * 0.8)
    train_set = trainset[:split_idx]
    val_set = trainset[split_idx:]

    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")

    # Initialize GEPA optimizer
    optimizer = GEPA(
        metric=metric,
        auto="light",
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=lm_teacher
    )

    # Run optimization
    print("\nRunning GEPA optimization...")
    optimized_generator = optimizer.compile(
        generator,
        trainset=train_set,
        valset=val_set
    )

    # Save optimized model
    optimized_generator.save(output_path)
    print(f"\nOptimized model saved to: {output_path}")

    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATION ON VALIDATION SET")
    print("="*60)

    evaluate = dspy.Evaluate(
        devset=val_set,
        metric=metric,
        num_threads=num_threads,
        display_progress=True
    )

    results = evaluate(optimized_generator)
    print(f"\nFinal validation score: {results}")

    return optimized_generator


def main():
    parser = argparse.ArgumentParser(description="Optimize Response Generation with GEPA")
    parser.add_argument(
        "--reference-file",
        type=str,
        default="human/generation_tasks/RAG_clapnq.jsonl",
        help="Path to reference JSONL file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_generator.json",
        help="Output path for optimized model"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("="*60)
    print("RESPONSE GENERATION OPTIMIZATION")
    print("="*60)
    print(f"Reference file: {args.reference_file}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")

    # Load and sample data
    print("\n[1/2] Loading reference data...")
    data = load_reference_data(args.reference_file)
    print(f"Loaded {len(data)} entries with agent responses")

    print("\n[2/2] Creating stratified sample...")
    samples = stratified_sample(data, n_samples=args.samples, seed=args.seed)

    # Run optimization
    optimized = run_optimization(
        samples=samples,
        output_path=args.output,
        num_threads=args.threads
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
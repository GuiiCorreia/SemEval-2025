"""
Unified Classification Module Optimization Script

This script uses DSPy GEPA optimizer to improve the UnifiedClassifier
which classifies both Answerability and Question Type.

Requires pre-built datasets from build_optimization_dataset.py.

Usage:
    python optimize_classification.py --train-file optimization_data/train.jsonl --val-file optimization_data/val.jsonl
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, field

import dspy
from dspy import GEPA
from dotenv import load_dotenv

from src.dspy_classification import (
    UnifiedClassifier,
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

STUDENT_MODEL = "gemini/gemini-3-flash-preview"
TEACHER_MODEL = "gemini/gemini-3-flash-preview"

STUDENT_TEMPERATURE = 0.2
TEACHER_TEMPERATURE = 0.2

MAX_TOKENS = 2000

# Create LM instances
lm_student = dspy.LM(STUDENT_MODEL, temperature=STUDENT_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=GEMINI_API_KEY, cache=False)
lm_teacher = dspy.LM(TEACHER_MODEL, temperature=TEACHER_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=GEMINI_API_KEY, cache=False)

# Configure default LM
dspy.configure(lm=lm_student)


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ClassificationSample:
    """A training sample for unified classification optimization."""
    task_id: str
    collection: str
    question: str
    conversation_history: List[Dict[str, str]]
    documents: List[Dict[str, Any]]
    ground_truth_answerability: str
    ground_truth_question_types: List[str] = field(default_factory=list)


def load_dataset(filepath: str) -> List[ClassificationSample]:
    """Load pre-built dataset from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            sample = ClassificationSample(
                task_id=entry.get('task_id', ''),
                collection=entry.get('collection', ''),
                question=entry.get('question', ''),
                conversation_history=entry.get('conversation_history', []),
                documents=entry.get('documents', []),
                ground_truth_answerability=entry.get('answerability', ''),
                ground_truth_question_types=entry.get('question_types', [])
            )
            samples.append(sample)
    return samples


def print_dataset_distribution(samples: List[ClassificationSample], label: str):
    """Print distribution summary for a dataset."""
    from collections import defaultdict

    print(f"\n{label}: {len(samples)} samples")

    # By Collection
    coll_dist = defaultdict(int)
    for s in samples:
        coll_dist[s.collection] += 1
    print(f"  By Collection: {dict(sorted(coll_dist.items()))}")

    # By Answerability
    ans_dist = defaultdict(int)
    for s in samples:
        ans_dist[s.ground_truth_answerability] += 1
    print(f"  By Answerability: {dict(sorted(ans_dist.items()))}")

    # By Question Type
    qt_dist = defaultdict(int)
    for s in samples:
        for qt in s.ground_truth_question_types:
            qt_dist[qt] += 1
    print(f"  By Question Type: {dict(sorted(qt_dist.items()))}")


# =============================================================================
# METRIC FOR UNIFIED CLASSIFICATION
# =============================================================================

def compute_f1_multilabel(predicted: List[str], ground_truth: List[str]) -> float:
    """Compute F1 score for multi-label classification."""
    pred_set = set(t.lower() for t in predicted)
    truth_set = set(t.lower() for t in ground_truth)

    if not pred_set and not truth_set:
        return 1.0
    if not pred_set or not truth_set:
        return 0.0

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(truth_set) if truth_set else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def unified_classification_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Evaluate unified classification (answerability + question type).

    Score is weighted average:
    - Answerability: 50% (exact match)
    - Question Type: 50% (F1 score for multi-label)
    """
    feedback_parts = []

    # === Answerability Evaluation ===
    gt_answerability = example.ground_truth_answerability
    pred_answerability = getattr(prediction, 'answerability', None)

    if pred_answerability is None:
        answerability_score = 0
        feedback_parts.append(f"Answerability: No prediction. Expected: {gt_answerability}")
    else:
        gt_norm = gt_answerability.upper().strip()
        pred_norm = pred_answerability.upper().strip()
        answerability_correct = gt_norm == pred_norm
        answerability_score = 1 if answerability_correct else 0

        if answerability_correct:
            feedback_parts.append(f"Answerability: Correct ({pred_answerability})")
        else:
            feedback_parts.append(f"Answerability: Incorrect. Predicted '{pred_answerability}', expected '{gt_answerability}'")

    # === Question Type Evaluation (Multi-label F1) ===
    gt_question_types = example.ground_truth_question_types
    pred_question_types = getattr(prediction, 'question_types', [])

    if isinstance(pred_question_types, str):
        pred_question_types = [t.strip() for t in pred_question_types.split(',') if t.strip()]

    question_type_f1 = compute_f1_multilabel(pred_question_types, gt_question_types)

    gt_types_str = ', '.join(gt_question_types) if gt_question_types else 'None'
    pred_types_str = ', '.join(pred_question_types) if pred_question_types else 'None'
    feedback_parts.append(f"Question Types: F1={question_type_f1:.2f}. Predicted [{pred_types_str}], expected [{gt_types_str}]")

    # === Combined Score ===
    combined_score = 0.5 * answerability_score + 0.5 * question_type_f1

    feedback = "\n".join(feedback_parts)

    return dspy.Prediction(
        score=combined_score,
        feedback=feedback,
        answerability_score=answerability_score,
        question_type_f1=question_type_f1
    )


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def _samples_to_examples(samples: List[ClassificationSample]) -> List[dspy.Example]:
    """Convert ClassificationSamples to dspy.Examples."""
    examples = []
    for sample in samples:
        conv_history = format_conversation_history(sample.conversation_history)
        docs = format_documents(sample.documents)

        ex = dspy.Example(
            conversation_history=conv_history,
            current_question=sample.question,
            retrieved_documents=docs,
            ground_truth_answerability=sample.ground_truth_answerability,
            ground_truth_question_types=sample.ground_truth_question_types,
            task_id=sample.task_id
        ).with_inputs('conversation_history', 'current_question', 'retrieved_documents')
        examples.append(ex)
    return examples


def run_optimization(
    train_samples: List[ClassificationSample],
    val_samples: List[ClassificationSample],
    output_path: str = "optimized_classifier.json",
    num_threads: int = 4
):
    """
    Run GEPA optimization on the unified classifier (answerability + question type).
    """
    print("\n" + "="*60)
    print("STARTING UNIFIED CLASSIFICATION OPTIMIZATION")
    print("="*60)

    # Create the unified classifier module
    classifier = UnifiedClassifier()

    # Convert samples to dspy.Example format
    train_set = _samples_to_examples(train_samples)
    val_set = _samples_to_examples(val_samples)

    print(f"Training set: {len(train_set)} examples")
    print(f"Validation set: {len(val_set)} examples")

    # Initialize GEPA optimizer with unified metric
    optimizer = GEPA(
        metric=unified_classification_metric,
        auto="light",
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=lm_teacher
    )

    # Run optimization
    print("\nRunning GEPA optimization...")
    optimized_classifier = optimizer.compile(
        classifier,
        trainset=train_set,
        valset=val_set
    )

    # Save optimized model
    optimized_classifier.save(output_path)
    print(f"\nOptimized model saved to: {output_path}")

    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATION ON VALIDATION SET")
    print("="*60)

    evaluate = dspy.Evaluate(
        devset=val_set,
        metric=unified_classification_metric,
        num_threads=num_threads,
        display_progress=True
    )

    results = evaluate(optimized_classifier)
    print(f"\nFinal validation score (50% answerability + 50% question type F1): {results}")

    return optimized_classifier


def main():
    parser = argparse.ArgumentParser(description="Optimize Classification with GEPA")
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training dataset JSONL (from build_optimization_dataset.py)"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        required=True,
        help="Path to validation dataset JSONL (from build_optimization_dataset.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_classifier.json",
        help="Output path for optimized model"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads"
    )

    args = parser.parse_args()

    print("="*60)
    print("UNIFIED CLASSIFICATION OPTIMIZATION")
    print("="*60)
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Output: {args.output}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Teacher: {TEACHER_MODEL}")

    # Load pre-built datasets
    print("\n[1/2] Loading datasets...")
    train_samples = load_dataset(args.train_file)
    val_samples = load_dataset(args.val_file)

    print_dataset_distribution(train_samples, "Train set")
    print_dataset_distribution(val_samples, "Val set")

    # Run optimization
    print("\n[2/2] Running optimization...")
    optimized = run_optimization(
        train_samples=train_samples,
        val_samples=val_samples,
        output_path=args.output,
        num_threads=args.threads
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
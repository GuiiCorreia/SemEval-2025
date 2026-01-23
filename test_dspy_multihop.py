#!/usr/bin/env python3
"""
Test Script for DSPy Multi-Hop RAG Pipeline

Evaluates the multi-hop retriever using the validation dataset
and DSPy's Evaluate functionality.

Usage:
    python test_dspy_multihop.py --val-file dataset/optimization_data/val.jsonl
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

import dspy
from dotenv import load_dotenv

from src.config import load_config
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval import HybridRetriever
from src.dspy_multihop import create_multihop_retriever

# Load environment variables
load_dotenv()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL = "openai/openai/gpt-oss-120b"
TEMPERATURE = 0.4
MAX_TOKENS = 10000


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ValidationSample:
    """A validation sample for evaluation."""
    task_id: str
    collection: str
    question: str
    conversation_history: List[Dict[str, str]]
    reference_doc_ids: List[str]
    reference_docs: List[Dict[str, Any]]
    answerability: str
    question_types: List[str]


def load_validation_data(filepath: str, limit: int = None) -> List[ValidationSample]:
    """Load validation dataset from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            entry = json.loads(line)
            sample = ValidationSample(
                task_id=entry.get('task_id', ''),
                collection=entry.get('collection', ''),
                question=entry.get('question', ''),
                conversation_history=entry.get('conversation_history', []),
                reference_doc_ids=entry.get('reference_doc_ids', []),
                reference_docs=entry.get('documents', []),
                answerability=entry.get('answerability', ''),
                question_types=entry.get('question_types', [])
            )
            samples.append(sample)
    return samples


# =============================================================================
# LLM JUDGE FOR RETRIEVAL EVALUATION
# =============================================================================

class RetrievalJudge(dspy.Signature):
    """Judge whether retrieved documents match the reference documents for answering the question."""

    question = dspy.InputField(desc="The user's question")
    reference_documents = dspy.InputField(desc="The ground truth reference documents")
    retrieved_documents = dspy.InputField(desc="The documents retrieved by the system")

    match_analysis = dspy.OutputField(desc="Analysis of how well retrieved docs match reference docs")
    verdict = dspy.OutputField(desc="PASS if retrieved docs contain the key information from references, FAIL otherwise")
    feedback = dspy.OutputField(desc="Specific suggestions for better search queries to retrieve the reference documents")


class LLMJudge(dspy.Module):
    """LLM-based judge for evaluating retrieval quality."""

    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RetrievalJudge)

    def forward(
        self,
        question: str,
        reference_docs: List[Dict[str, Any]],
        retrieved_docs: List[Dict[str, Any]]
    ) -> dspy.Prediction:
        # Format reference documents
        ref_str = self._format_docs(reference_docs, "Reference")

        # Format retrieved documents
        ret_str = self._format_docs(retrieved_docs, "Retrieved")

        result = self.judge(
            question=question,
            reference_documents=ref_str,
            retrieved_documents=ret_str
        )

        # Parse verdict
        verdict_lower = result.verdict.lower()
        passed = 'pass' in verdict_lower and 'fail' not in verdict_lower

        return dspy.Prediction(
            match_analysis=result.match_analysis,
            verdict=result.verdict,
            feedback=result.feedback,
            passed=passed
        )

    def _format_docs(self, docs: List[Dict[str, Any]], label: str) -> str:
        """Format documents for the judge prompt."""
        if not docs:
            return f"No {label.lower()} documents."

        formatted = []
        for i, doc in enumerate(docs, 1):
            doc_id = doc.get('document_id', 'N/A')
            title = doc.get('title', 'Untitled')
            text = doc.get('text', '')[:5000]
            formatted.append(f"[{label} {i}] ID: {doc_id}\nTitle: {title}\n{text}")

        return "\n\n".join(formatted)


# Global judge instance (created after DSPy is configured)
llm_judge = None


# =============================================================================
# METRICS
# =============================================================================

def retrieval_recall_metric(example, prediction, trace=None):
    """
    Calculate retrieval recall: how many reference docs were retrieved.

    Returns score between 0.0 and 1.0.
    """
    # Get reference doc IDs from example
    reference_ids = set(example.reference_doc_ids)

    # If no reference docs, consider it a pass (nothing to retrieve)
    if not reference_ids:
        return 1.0

    # Get retrieved doc IDs from prediction
    retrieved_docs = prediction.final_documents if hasattr(prediction, 'final_documents') else []
    retrieved_ids = {doc.get('document_id') for doc in retrieved_docs if doc.get('document_id')}

    # Calculate recall
    overlap = retrieved_ids & reference_ids
    recall = len(overlap) / len(reference_ids)

    return recall


def llm_judge_metric(example, prediction, trace=None):
    """
    Use LLM Judge to evaluate retrieval quality.

    Returns 1.0 if PASS, 0.0 if FAIL.
    """
    global llm_judge

    # Get reference docs and retrieved docs
    reference_docs = example.reference_docs if hasattr(example, 'reference_docs') else []
    retrieved_docs = prediction.final_documents if hasattr(prediction, 'final_documents') else []

    # If no reference docs, consider it a pass
    if not reference_docs:
        return 1.0

    # If no retrieved docs, it's a fail
    if not retrieved_docs:
        return 0.0

    try:
        # Run judge
        result = llm_judge(
            question=example.question,
            reference_docs=reference_docs,
            retrieved_docs=retrieved_docs
        )
        return 1.0 if result.passed else 0.0
    except Exception as e:
        print(f"Judge error: {e}")
        return 0.0


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    val_samples: List[ValidationSample],
    hybrid_retriever: HybridRetriever,
    num_hops: int = 3,
    num_threads: int = 1,
    use_llm_judge: bool = True,
    optimized_model_path: str = None
):
    """Run evaluation on validation samples."""
    global llm_judge

    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)

    # Initialize LLM Judge if needed
    if use_llm_judge:
        print("Initializing LLM Judge...")
        llm_judge = LLMJudge()

    # Convert samples to dspy.Example format
    eval_set = []
    for sample in val_samples:
        ex = dspy.Example(
            question=sample.question,
            conversation_history=sample.conversation_history,
            reference_doc_ids=sample.reference_doc_ids,
            reference_docs=sample.reference_docs,
            collection=sample.collection,
            task_id=sample.task_id
        ).with_inputs('question', 'conversation_history')
        eval_set.append(ex)

    print(f"Evaluation set: {len(eval_set)} samples")

    # Group samples by collection
    collections = {}
    for sample in val_samples:
        if sample.collection not in collections:
            collections[sample.collection] = 0
        collections[sample.collection] += 1

    print("Samples by collection:")
    for coll, count in sorted(collections.items()):
        print(f"  {coll}: {count}")

    # Create multi-hop retriever for each collection and evaluate
    all_results = []

    for collection_name in collections.keys():
        print(f"\n--- Evaluating collection: {collection_name} ---")

        # Filter samples for this collection
        collection_samples = [ex for ex in eval_set if ex.collection == collection_name]
        print(f"Samples for this collection: {len(collection_samples)}")

        # Create retriever for this collection
        retriever = create_multihop_retriever(
            hybrid_retriever=hybrid_retriever,
            collection_name=collection_name,
            num_hops=num_hops
        )

        # Load optimized model if provided
        if optimized_model_path:
            print(f"Loading optimized model from: {optimized_model_path}")
            retriever.load(optimized_model_path)

        # Run evaluation with Recall metric
        print("\nRunning Recall evaluation...")
        evaluate_recall = dspy.Evaluate(
            devset=collection_samples,
            metric=retrieval_recall_metric,
            num_threads=num_threads,
            display_progress=True
        )
        recall_result = evaluate_recall(retriever)
        recall_score = float(recall_result.score) if hasattr(recall_result, 'score') else float(recall_result)
        print(f"Recall Score: {recall_score:.4f}")

        # Run evaluation with LLM Judge metric
        judge_score = 0.0
        if use_llm_judge:
            print("\nRunning LLM Judge evaluation...")
            evaluate_judge = dspy.Evaluate(
                devset=collection_samples,
                metric=llm_judge_metric,
                num_threads=num_threads,
                display_progress=True
            )
            judge_result = evaluate_judge(retriever)
            judge_score = float(judge_result.score) if hasattr(judge_result, 'score') else float(judge_result)
            print(f"LLM Judge Score: {judge_score:.4f}")

        all_results.append({
            'collection': collection_name,

            'num_samples': len(collection_samples),
            'recall': recall_score,
            'judge': judge_score
        })

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    total_samples = sum(r['num_samples'] for r in all_results)
    weighted_recall = sum(r['recall'] * r['num_samples'] for r in all_results) / total_samples
    weighted_judge = sum(r['judge'] * r['num_samples'] for r in all_results) / total_samples

    print(f"\n{'Collection':<50} {'Samples':>8} {'Recall':>10} {'Judge':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['collection']:<50} {r['num_samples']:>8} {r['recall']:>10.4f} {r['judge']:>10.4f}")
    print("-" * 80)
    print(f"{'WEIGHTED AVERAGE':<50} {total_samples:>8} {weighted_recall:>10.4f} {weighted_judge:>10.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test DSPy Multi-Hop RAG Pipeline")
    parser.add_argument(
        "--val-file",
        type=str,
        default="dataset/optimization_data/val.jsonl",
        help="Path to validation dataset JSONL"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=3,
        help="Number of retrieval hops"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads"
    )
    parser.add_argument(
        "--load-optimized",
        type=str,
        default=None,
        help="Path to optimized model JSON (e.g., optimized_multihop.json)"
    )

    args = parser.parse_args()

    print("="*60)
    print("DSPy MULTI-HOP RAG EVALUATION")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Base URL: {DEEPINFRA_BASE_URL}")
    print(f"Val file: {args.val_file}")
    print(f"Num hops: {args.num_hops}")
    print(f"Threads: {args.threads}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    if args.load_optimized:
        print(f"Loading optimized model: {args.load_optimized}")

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Configure DSPy
    print("\n[1/3] Configuring DSPy...")
    lm = dspy.LM(
        MODEL,
        api_key=api_key,
        base_url=DEEPINFRA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        cache=False
    )
    dspy.configure(lm=lm)

    # Initialize retrieval infrastructure
    print("\n[2/3] Initializing retrieval infrastructure...")
    config = load_config()
    embedding_service = EmbeddingService(config.embedding)
    vector_store = VectorStore(config.qdrant, embedding_service)
    hybrid_retriever = HybridRetriever(config.retrieval, vector_store)

    # Load validation data
    print("\n[3/3] Loading validation data...")
    val_samples = load_validation_data(args.val_file, limit=args.limit)
    print(f"Loaded {len(val_samples)} validation samples")

    # Run evaluation
    results = run_evaluation(
        val_samples=val_samples,
        hybrid_retriever=hybrid_retriever,
        num_hops=args.num_hops,
        num_threads=args.threads,
        optimized_model_path=args.load_optimized
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
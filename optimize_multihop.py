"""
Multi-Hop RAG Optimization Script

This script uses DSPy GEPA optimizer to improve the multi-hop query generation
and notes building modules using an LLM judge for feedback.

Requires pre-built datasets from build_optimization_dataset.py.

Usage:
    python optimize_multihop.py --train-file optimization_data/train.jsonl --val-file optimization_data/val.jsonl
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

import dspy
from dspy import GEPA
from dotenv import load_dotenv
import mlflow

# Configure MLflow tracking
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=True
)
mlflow.set_experiment("multihop-rag-optimization")

from src.config import load_config
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval import HybridRetriever
from src.dspy_multihop import create_multihop_retriever

# Load environment variables
load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# DeepInfra OpenAI-compatible endpoint
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

STUDENT_MODEL = "openai/openai/gpt-oss-120b"
TEACHER_MODEL = "openai/openai/gpt-oss-120b"
JUDGE_MODEL = "openai/openai/gpt-oss-120b"

STUDENT_TEMPERATURE = 0.4
TEACHER_TEMPERATURE = 0.4
JUDGE_TEMPERATURE = 0.2

MAX_TOKENS = 10000

# Create LM instances (cache disabled for optimization)
lm_student = dspy.LM(STUDENT_MODEL, temperature=STUDENT_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=OPENAI_API_KEY, base_url=DEEPINFRA_BASE_URL, cache=False)
lm_teacher = dspy.LM(TEACHER_MODEL, temperature=TEACHER_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=OPENAI_API_KEY, base_url=DEEPINFRA_BASE_URL, cache=False)
lm_judge = dspy.LM(JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=OPENAI_API_KEY, base_url=DEEPINFRA_BASE_URL, cache=False)

# Configure default LM (student)
dspy.configure(lm=lm_student)


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class TrainingSample:
    """A training sample for optimization."""
    task_id: str
    collection: str
    question: str
    conversation_history: List[Dict[str, str]]
    reference_doc_ids: List[str]
    reference_docs: List[Dict[str, Any]]
    answerability: str
    question_types: List[str]
    ground_truth_response: str = ""


# Collection descriptions for context
COLLECTION_INFO = {
    "mt-rag-clapnq-elser-512-100-20240503": {
        "name": "ClapNQ",
        "domain": "General knowledge (Wikipedia-based)",
        "description": "Questions based on Wikipedia articles covering diverse topics"
    },
    "mt-rag-fiqa-beir-elser-512-100-20240501": {
        "name": "FiQA",
        "domain": "Financial",
        "description": "Financial question answering from forums and financial documents"
    },
    "mt-rag-govt-elser-512-100-20240611": {
        "name": "Government",
        "domain": "Government services",
        "description": "Questions about government services, policies, and procedures"
    },
    "mt-rag-ibmcloud-elser-512-100-20240502": {
        "name": "IBM Cloud",
        "domain": "Technical documentation",
        "description": "Technical questions about IBM Cloud services and documentation"
    }
}


def load_dataset(filepath: str) -> List[TrainingSample]:
    """Load pre-built dataset from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            sample = TrainingSample(
                task_id=entry.get('task_id', ''),
                collection=entry.get('collection', ''),
                question=entry.get('question', ''),
                conversation_history=entry.get('conversation_history', []),
                reference_doc_ids=entry.get('reference_doc_ids', []),
                reference_docs=entry.get('documents', []),
                answerability=entry.get('answerability', 'UNKNOWN'),
                question_types=entry.get('question_types', []),
                ground_truth_response=entry.get('ground_truth_response', '')
            )
            samples.append(sample)
    return samples


def print_dataset_distribution(samples: List[TrainingSample], label: str):
    """Print distribution summary for a dataset."""
    print(f"\n{label}: {len(samples)} samples")

    # By Collection
    coll_dist = defaultdict(int)
    for s in samples:
        coll_dist[s.collection] += 1
    print(f"  By Collection: {dict(sorted(coll_dist.items()))}")


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


# =============================================================================
# METRIC WITH FEEDBACK FOR GEPA
# =============================================================================

def create_metric_with_feedback():
    """Create a metric function with feedback for GEPA optimization."""

    def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
        # Extract retrieved documents
        retrieved_docs = prediction.final_documents if hasattr(prediction, 'final_documents') else []
        retrieved_ids = {doc.get('document_id') for doc in retrieved_docs if doc.get('document_id')}

        # Get reference info from example
        reference_docs = example.reference_docs
        reference_ids = set(example.reference_doc_ids)
        question = example.question

        # Get question dimensions
        collection = getattr(example, 'collection', 'unknown')
        answerability = getattr(example, 'answerability', 'UNKNOWN')
        question_types = getattr(example, 'question_types', [])
        conversation_history = getattr(example, 'conversation_history', [])
        ground_truth_response = getattr(example, 'ground_truth_response', '')

        # Get collection info
        coll_info = COLLECTION_INFO.get(collection, {})
        dataset_name = coll_info.get('name', collection)
        dataset_domain = coll_info.get('domain', 'Unknown')

        # Calculate recall
        overlap = retrieved_ids & reference_ids
        missing_ids = reference_ids - retrieved_ids
        recall = len(overlap) / len(reference_ids) if reference_ids else 1.0

        # Determine if multi-turn
        is_multi_turn = len(conversation_history) > 0
        turn_number = len(conversation_history) // 2 + 1

        # Build structured feedback
        feedback_parts = [
            "QUESTION DIMENSIONS:",
            f"  Dataset: {dataset_name} ({dataset_domain})",
            f"  Question Types: {', '.join(question_types) if question_types else 'Not specified'}",
            f"  Answerability: {answerability}",
            f"  Turn: {turn_number} ({'Multi-turn' if is_multi_turn else 'First turn'})",
            "",
            "QUESTION:",
            f"  {question}",
            "",
            "EXPECTED RESPONSE (Ground Truth):",
            f"  {ground_truth_response[:2000] if ground_truth_response else 'Not available'}",
            "",
            "RETRIEVAL RESULT:",
            f"  Recall: {recall:.1%} ({len(overlap)}/{len(reference_ids)} documents)",
        ]

        if missing_ids:
            feedback_parts.append("")
            feedback_parts.append(f"MISSING DOCUMENTS ({len(missing_ids)}):")

            for doc in reference_docs:
                doc_id = doc.get('document_id', '')
                if doc_id in missing_ids:
                    title = doc.get('title', 'Untitled')
                    text_preview = doc.get('text', '')[:3000]
                    feedback_parts.append(f"\n  [{doc_id}]")
                    feedback_parts.append(f"  Title: {title}")
                    feedback_parts.append(f"  Content: {text_preview}...")
        else:
            feedback_parts.append("")
            feedback_parts.append("All reference documents retrieved successfully.")

        feedback = "\n".join(feedback_parts)
        return dspy.Prediction(score=recall, feedback=feedback)

    return metric_with_feedback


# =============================================================================
# MAIN OPTIMIZATION PIPELINE
# =============================================================================

def run_optimization(
    train_samples: List[TrainingSample],
    val_samples: List[TrainingSample],
    hybrid_retriever: HybridRetriever,
    collection_name: str,
    output_path: str = "optimized_multihop.json",
    num_threads: int = 4
):
    """Run GEPA optimization on the multi-hop retriever."""
    print("\n" + "="*60)
    print("STARTING GEPA OPTIMIZATION")
    print("="*60)

    # Create the multi-hop retriever module
    retriever = create_multihop_retriever(
        hybrid_retriever=hybrid_retriever,
        collection_name=collection_name,
        num_hops=3
    )

    # Create metric
    metric = create_metric_with_feedback()

    # Convert samples to dspy.Example format
    train_set = []
    for sample in train_samples:
        ex = dspy.Example(
            question=sample.question,
            conversation_history=sample.conversation_history,
            reference_doc_ids=sample.reference_doc_ids,
            reference_docs=sample.reference_docs,
            task_id=sample.task_id,
            collection=sample.collection,
            answerability=sample.answerability,
            question_types=sample.question_types,
            ground_truth_response=sample.ground_truth_response
        ).with_inputs('question', 'conversation_history')
        train_set.append(ex)

    val_set = []
    for sample in val_samples:
        ex = dspy.Example(
            question=sample.question,
            conversation_history=sample.conversation_history,
            reference_doc_ids=sample.reference_doc_ids,
            reference_docs=sample.reference_docs,
            task_id=sample.task_id,
            collection=sample.collection,
            answerability=sample.answerability,
            question_types=sample.question_types,
            ground_truth_response=sample.ground_truth_response
        ).with_inputs('question', 'conversation_history')
        val_set.append(ex)

    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")

    # Initialize GEPA optimizer
    optimizer = GEPA(
        metric=metric,
        auto="light",
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=lm_teacher,
        use_mlflow=True
    )

    # Run optimization
    print("\nRunning GEPA optimization...")
    optimized_retriever = optimizer.compile(
        retriever,
        trainset=train_set,
        valset=val_set
    )

    # Save optimized model
    optimized_retriever.save(output_path)
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

    results = evaluate(optimized_retriever)
    print(f"\nFinal validation score: {results}")

    return optimized_retriever


def main():
    parser = argparse.ArgumentParser(description="Optimize Multi-Hop RAG with GEPA")
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
        "--collection",
        type=str,
        default=None,
        help="Specific collection to use (default: most common from samples)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_multihop.json",
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
    print("MULTI-HOP RAG OPTIMIZATION")
    print("="*60)
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Output: {args.output}")
    print(f"Base URL: {DEEPINFRA_BASE_URL}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")

    # Load pre-built datasets
    print("\n[1/3] Loading datasets...")
    train_samples = load_dataset(args.train_file)
    val_samples = load_dataset(args.val_file)

    print_dataset_distribution(train_samples, "Train set")
    print_dataset_distribution(val_samples, "Val set")

    # Setup retrieval infrastructure
    print("\n[2/3] Initializing retrieval infrastructure...")
    config = load_config()
    embedding_service = EmbeddingService(config.embedding)
    vector_store = VectorStore(config.qdrant, embedding_service)
    hybrid_retriever = HybridRetriever(config.retrieval, vector_store)

    # Determine collection to use
    if args.collection:
        collection_name = args.collection
    else:
        # Use the most common collection from samples
        all_samples = train_samples + val_samples
        collections = [s.collection for s in all_samples]
        collection_name = max(set(collections), key=collections.count)

    print(f"Using collection: {collection_name}")

    # Run optimization
    print("\n[3/3] Running optimization...")
    optimized = run_optimization(
        train_samples=train_samples,
        val_samples=val_samples,
        hybrid_retriever=hybrid_retriever,
        collection_name=collection_name,
        output_path=args.output,
        num_threads=args.threads
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
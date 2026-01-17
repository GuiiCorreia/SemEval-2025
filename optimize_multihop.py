"""
Multi-Hop RAG Optimization Script

This script uses DSPy GEPA optimizer to improve the multi-hop query generation
and notes building modules using an LLM judge for feedback.

Usage:
    python optimize_multihop.py --collection <collection_name> --samples 100
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

import dspy
from dspy import GEPA
from dotenv import load_dotenv

from src.config import load_config
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval import HybridRetriever
from src.dspy_multihop import MultiHopRetriever, create_multihop_retriever

# Load environment variables
load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Models - easy to change here
# For Gemini: "gemini/gemini-2.5-flash-preview", "gemini/gemini-2.5-pro-preview"
# For Groq: "groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile"
STUDENT_MODEL = "gemini/gemini-3-flash-preview"
TEACHER_MODEL = "gemini/gemini-3-flash-preview"
JUDGE_MODEL = "gemini/gemini-3-flash-preview"

# API key to use (change if using Groq models)
API_KEY = GEMINI_API_KEY  # or GROQ_API_KEY

# Temperature settings
STUDENT_TEMPERATURE = 0.4
TEACHER_TEMPERATURE = 0.4
JUDGE_TEMPERATURE = 0.2

# Max tokens
MAX_TOKENS = 8000


# Create LM instances (cache disabled for optimization)
lm_student = dspy.LM(STUDENT_MODEL, temperature=STUDENT_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=API_KEY, cache=False)
lm_teacher = dspy.LM(TEACHER_MODEL, temperature=TEACHER_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=API_KEY, cache=False)
lm_judge = dspy.LM(JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, max_tokens=MAX_TOKENS, api_key=API_KEY, cache=False)

# Configure default LM (student)
dspy.configure(lm=lm_student)


# =============================================================================
# DATA LOADING AND STRATIFIED SAMPLING
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


def load_reference_data(filepath: str) -> List[Dict[str, Any]]:
    """Load reference data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # Only include entries that have reference contexts
            if entry.get('contexts'):
                data.append(entry)
    return data


def stratified_sample(
    data: List[Dict[str, Any]],
    n_samples: int = 100,
    stratify_field: str = 'Collection',
    seed: int = 42
) -> List[TrainingSample]:
    """
    Create stratified sample balanced by collection.

    Args:
        data: Full dataset
        n_samples: Total number of samples to select
        stratify_field: Field to stratify by
        seed: Random seed for reproducibility

    Returns:
        List of TrainingSample objects
    """
    random.seed(seed)

    # Group by stratification field
    groups = defaultdict(list)
    for entry in data:
        key = entry.get(stratify_field, 'unknown')
        groups[key].append(entry)

    # Calculate samples per group
    n_groups = len(groups)
    base_per_group = n_samples // n_groups
    remainder = n_samples % n_groups

    print(f"Stratified sampling: {n_samples} samples across {n_groups} groups")
    for group_name, group_data in groups.items():
        print(f"  {group_name}: {len(group_data)} available")

    # Sample from each group
    samples = []
    group_names = sorted(groups.keys())

    for i, group_name in enumerate(group_names):
        group_data = groups[group_name]
        # Distribute remainder to first groups
        n_for_group = base_per_group + (1 if i < remainder else 0)
        n_for_group = min(n_for_group, len(group_data))

        selected = random.sample(group_data, n_for_group)

        for entry in selected:
            # Extract question from last user input
            inputs = entry.get('input', [])
            question = inputs[-1].get('text', '') if inputs else ''

            # Build conversation history (all but last message)
            conversation_history = inputs[:-1] if len(inputs) > 1 else []

            # Extract reference document IDs
            contexts = entry.get('contexts', [])
            ref_doc_ids = [ctx.get('document_id') for ctx in contexts if ctx.get('document_id')]

            sample = TrainingSample(
                task_id=entry.get('task_id', ''),
                collection=entry.get('Collection', ''),
                question=question,
                conversation_history=conversation_history,
                reference_doc_ids=ref_doc_ids,
                reference_docs=contexts
            )
            samples.append(sample)

    print(f"Total samples selected: {len(samples)}")
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
        """
        Judge retrieval quality.

        Args:
            question: The user's question
            reference_docs: Ground truth documents
            retrieved_docs: System-retrieved documents

        Returns:
            Prediction with verdict and feedback
        """
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
            text = doc.get('text', '')[:500]
            formatted.append(f"[{label} {i}] ID: {doc_id}\nTitle: {title}\n{text}")

        return "\n\n".join(formatted)


def compute_doc_overlap(
    retrieved_doc_ids: List[str],
    reference_doc_ids: List[str]
) -> Dict[str, float]:
    """
    Compute document overlap metrics.

    Args:
        retrieved_doc_ids: IDs of retrieved documents
        reference_doc_ids: IDs of reference documents

    Returns:
        Dictionary with recall, precision, and overlap count
    """
    retrieved_set = set(retrieved_doc_ids)
    reference_set = set(reference_doc_ids)

    overlap = retrieved_set & reference_set
    overlap_count = len(overlap)

    recall = overlap_count / len(reference_set) if reference_set else 0.0
    precision = overlap_count / len(retrieved_set) if retrieved_set else 0.0

    return {
        'overlap_count': overlap_count,
        'recall': recall,
        'precision': precision,
        'retrieved_count': len(retrieved_set),
        'reference_count': len(reference_set)
    }


# =============================================================================
# METRIC WITH FEEDBACK FOR GEPA
# =============================================================================

def create_metric_with_feedback(judge: LLMJudge):
    """
    Create a metric function with feedback for GEPA optimization.

    Args:
        judge: LLMJudge instance

    Returns:
        Metric function compatible with GEPA
    """
    def metric_with_feedback(example, prediction, trace=None):
        """
        Evaluate retrieval using LLM judge and recall.

        Args:
            example: Training example with reference docs
            prediction: Model prediction with retrieved docs

        Returns:
            dspy.Prediction with score and feedback
        """
        # Extract retrieved documents
        retrieved_docs = prediction.final_documents if hasattr(prediction, 'final_documents') else []
        retrieved_ids = {doc.get('document_id') for doc in retrieved_docs if doc.get('document_id')}

        # Get reference info from example
        reference_docs = example.reference_docs
        reference_ids = set(example.reference_doc_ids)
        question = example.question

        # Calculate recall
        overlap = retrieved_ids & reference_ids
        recall = len(overlap) / len(reference_ids) if reference_ids else 0.0

        # Use LLM judge for evaluation (with judge LM - lower temperature)
        with dspy.context(lm=lm_judge):
            judge_result = judge(
                question=question,
                reference_docs=reference_docs,
                retrieved_docs=retrieved_docs
            )

        # Score based on judge verdict
        judge_score = 1.0 if judge_result.passed else 0.0

        # Build feedback with recall info
        feedback = f"Recall: {len(overlap)}/{len(reference_ids)} ({recall:.1%})\n{judge_result.feedback}"

        return dspy.Prediction(score=judge_score, feedback=feedback, recall=recall)

    return metric_with_feedback


# =============================================================================
# MAIN OPTIMIZATION PIPELINE
# =============================================================================

def run_optimization(
    samples: List[TrainingSample],
    hybrid_retriever: HybridRetriever,
    collection_name: str,
    output_path: str = "optimized_multihop.json",
    num_threads: int = 4
):
    """
    Run GEPA optimization on the multi-hop retriever.

    Args:
        samples: Training samples
        hybrid_retriever: HybridRetriever instance
        collection_name: Collection to search
        output_path: Path to save optimized model
        num_threads: Number of parallel threads
    """
    print("\n" + "="*60)
    print("STARTING GEPA OPTIMIZATION")
    print("="*60)

    # Create the multi-hop retriever module
    retriever = create_multihop_retriever(
        hybrid_retriever=hybrid_retriever,
        collection_name=collection_name,
        num_hops=3
    )

    # Create judge and metric
    judge = LLMJudge()
    metric = create_metric_with_feedback(judge)

    # Convert samples to dspy.Example format
    trainset = []
    for sample in samples:
        ex = dspy.Example(
            question=sample.question,
            conversation_history=sample.conversation_history,
            reference_doc_ids=sample.reference_doc_ids,
            reference_docs=sample.reference_docs,
            task_id=sample.task_id
        ).with_inputs('question', 'conversation_history')
        trainset.append(ex)

    # Split into train and validation
    random.shuffle(trainset)
    split_idx = int(len(trainset) * 0.8)
    train_set = trainset[:split_idx]
    val_set = trainset[split_idx:]

    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")

    # Initialize GEPA optimizer
    from dspy.teleprompt import GEPA

    optimizer = GEPA(
        metric=metric,
        train_set=train_set,
        val_set=val_set,
        teacher_settings=dict(lm=teacher_lm),
        num_threads=num_threads
    )

    # Run optimization
    print("\nRunning GEPA optimization...")
    optimized_retriever = optimizer.compile(retriever)

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
        "--reference-file",
        type=str,
        default="human/generation_tasks/reference.jsonl",
        help="Path to reference JSONL file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training samples"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Specific collection to use (default: use collection from each sample)"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("="*60)
    print("MULTI-HOP RAG OPTIMIZATION")
    print("="*60)
    print(f"Reference file: {args.reference_file}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Judge: {JUDGE_MODEL}")

    # Load and sample data
    print("\n[1/3] Loading reference data...")
    data = load_reference_data(args.reference_file)
    print(f"Loaded {len(data)} entries with contexts")

    print("\n[2/3] Creating stratified sample...")
    samples = stratified_sample(data, n_samples=args.samples, seed=args.seed)

    # Setup retrieval infrastructure
    print("\n[3/3] Initializing retrieval infrastructure...")
    config = load_config()
    embedding_service = EmbeddingService(config.embedding)
    vector_store = VectorStore(config.qdrant, embedding_service)
    hybrid_retriever = HybridRetriever(config.retrieval, vector_store)

    # Determine collection to use
    if args.collection:
        collection_name = args.collection
    else:
        # Use the most common collection from samples
        collections = [s.collection for s in samples]
        collection_name = max(set(collections), key=collections.count)

    print(f"Using collection: {collection_name}")

    # Map collection names to corpus-compatible names
    collection_mapping = {
        'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
        'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
        'mt-rag-govt-elser-512-100-20240611': 'govt',
        'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud'
    }

    corpus_name = collection_mapping.get(collection_name, collection_name)

    # Load BM25 index
    corpus_file = f"corpora/passage_level/{corpus_name}.jsonl"
    if os.path.exists(corpus_file):
        print(f"Loading BM25 index from {corpus_file}...")
        hybrid_retriever.index_corpus_bm25(corpus_name, corpus_file)
    else:
        print(f"Warning: Corpus file not found: {corpus_file}")

    # Run optimization
    optimized = run_optimization(
        samples=samples,
        hybrid_retriever=hybrid_retriever,
        collection_name=corpus_name,
        output_path=args.output,
        num_threads=args.threads
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

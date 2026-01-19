"""
Build Optimization Dataset Script

This script creates balanced training and validation datasets for all optimization pipelines.
The datasets are stratified by:
1. Collection (dataset source)
2. Answerability (ANSWERABLE, UNANSWERABLE, CONVERSATIONAL, PARTIAL)
3. Question Type (diversity maximization for multi-label types)

Usage:
    python build_optimization_dataset.py --reference-file human/generation_tasks/RAG_clapnq.jsonl --samples 100 --output-dir optimization_data/

Output:
    - optimization_data/train.jsonl
    - optimization_data/val.jsonl
    - optimization_data/metadata.json (distribution stats)
"""

import os
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizationSample:
    """A sample for optimization pipelines."""
    task_id: str
    collection: str
    question: str
    conversation_history: List[Dict[str, str]]
    documents: List[Dict[str, Any]]
    answerability: str
    question_types: List[str] = field(default_factory=list)
    reference_doc_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reference_data(filepath: str) -> List[Dict[str, Any]]:
    """Load reference data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # Only include entries that have Answerability label
            if entry.get('Answerability'):
                data.append(entry)
    return data


def _entry_to_sample(entry: Dict[str, Any]) -> OptimizationSample:
    """Convert a data entry to an OptimizationSample."""
    inputs = entry.get('input', [])
    question = inputs[-1].get('text', '') if inputs else ''
    conversation_history = inputs[:-1] if len(inputs) > 1 else []
    documents = entry.get('contexts', [])

    # Answerability (take first if list)
    answerability = entry.get('Answerability', ['UNKNOWN'])
    if isinstance(answerability, list):
        answerability = answerability[0] if answerability else 'UNKNOWN'

    # Question types
    question_types = entry.get('Question Type', [])
    if isinstance(question_types, str):
        question_types = [question_types]

    # Reference document IDs
    ref_doc_ids = [doc.get('document_id') for doc in documents if doc.get('document_id')]

    return OptimizationSample(
        task_id=entry.get('task_id', ''),
        collection=entry.get('Collection', 'unknown'),
        question=question,
        conversation_history=conversation_history,
        documents=documents,
        answerability=answerability,
        question_types=question_types,
        reference_doc_ids=ref_doc_ids
    )


# =============================================================================
# STRATIFIED SAMPLING WITH BALANCING
# =============================================================================

def _get_question_types(entry: Dict[str, Any]) -> Set[str]:
    """Extract question types from entry as a set."""
    qt = entry.get('Question Type', [])
    if isinstance(qt, str):
        qt = [qt]
    return set(qt)


def _select_with_question_type_diversity(
    candidates: List[Dict[str, Any]],
    n_to_select: int,
    current_type_counts: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Select samples prioritizing question type diversity.

    Samples that contain underrepresented question types are prioritized.
    """
    if n_to_select >= len(candidates):
        return candidates.copy()

    selected = []
    remaining = candidates.copy()

    while len(selected) < n_to_select and remaining:
        # Score each candidate by how much it adds underrepresented types
        best_score = -1
        best_idx = 0

        for idx, entry in enumerate(remaining):
            types = _get_question_types(entry)
            # Score = sum of (1 / (count + 1)) for each type in entry
            # Lower counts = higher contribution to score
            score = sum(1.0 / (current_type_counts.get(t, 0) + 1) for t in types)
            if not types:
                score = 0.1  # Small score for entries without types

            if score > best_score:
                best_score = score
                best_idx = idx

        # Select the best candidate
        chosen = remaining.pop(best_idx)
        selected.append(chosen)

        # Update type counts
        for t in _get_question_types(chosen):
            current_type_counts[t] = current_type_counts.get(t, 0) + 1

    return selected


def stratified_sample_with_split(
    data: List[Dict[str, Any]],
    n_samples: int = 100,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[OptimizationSample], List[OptimizationSample], Dict[str, Any]]:
    """
    Create stratified train/val split balanced by Collection, Answerability, and Question Type diversity.

    Stratification hierarchy:
    1. Collection (dataset): Balanced across all collections
    2. Answerability: Within each collection, balanced across answerability labels
    3. Question Type: Within each (collection, answerability) group, prioritize diversity

    Both train and validation sets will have balanced distribution.

    Args:
        data: Full dataset
        n_samples: Total number of samples to select
        val_ratio: Ratio of samples for validation (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, metadata_dict)
    """
    random.seed(seed)

    # Step 1: Group by (Collection, Answerability) pairs
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for entry in data:
        collection = entry.get('Collection', 'unknown')
        answerability = entry.get('Answerability', ['UNKNOWN'])
        if isinstance(answerability, list):
            answerability = answerability[0] if answerability else 'UNKNOWN'
        key = (collection, answerability)
        groups[key].append(entry)

    # Get unique collections and answerability labels
    collections = sorted(set(k[0] for k in groups.keys()))
    answerability_labels = sorted(set(k[1] for k in groups.keys()))

    print(f"Stratified sampling: {n_samples} samples")
    print(f"Train/Val split: {(1-val_ratio)*100:.0f}% / {val_ratio*100:.0f}%")
    print(f"\nCollections ({len(collections)}): {', '.join(collections)}")
    print(f"Answerability labels ({len(answerability_labels)}): {', '.join(answerability_labels)}")

    print(f"\nGroup sizes (Collection, Answerability):")
    for key in sorted(groups.keys()):
        print(f"  {key}: {len(groups[key])} available")

    # Step 2: Calculate samples per group (balanced allocation)
    n_groups = len(groups)
    base_per_group = n_samples // n_groups
    remainder = n_samples % n_groups

    # Allocate samples to each group
    group_allocations = {}
    sorted_keys = sorted(groups.keys())
    for i, key in enumerate(sorted_keys):
        allocation = base_per_group + (1 if i < remainder else 0)
        allocation = min(allocation, len(groups[key]))
        group_allocations[key] = allocation

    print(f"\nAllocated samples per group:")
    for key, alloc in sorted(group_allocations.items()):
        print(f"  {key}: {alloc}")

    # Step 3: Select samples from each group with question type diversity
    train_samples = []
    val_samples = []

    # Track question type counts for diversity selection
    train_type_counts: Dict[str, int] = defaultdict(int)
    val_type_counts: Dict[str, int] = defaultdict(int)

    for key in sorted_keys:
        collection, answerability = key
        group_data = groups[key].copy()
        random.shuffle(group_data)

        n_for_group = group_allocations[key]
        if n_for_group == 0:
            continue

        # Split allocation into train/val
        n_val = max(1, int(n_for_group * val_ratio)) if n_for_group > 1 else 0
        n_train = n_for_group - n_val

        # Select train samples with diversity
        train_entries = _select_with_question_type_diversity(
            group_data, n_train, train_type_counts
        )

        # Remove selected from candidates
        train_ids = set(id(e) for e in train_entries)
        remaining = [e for e in group_data if id(e) not in train_ids]

        # Select val samples with diversity
        val_entries = _select_with_question_type_diversity(
            remaining, n_val, val_type_counts
        )

        # Convert to samples
        for entry in train_entries:
            train_samples.append(_entry_to_sample(entry))
        for entry in val_entries:
            val_samples.append(_entry_to_sample(entry))

    # Shuffle both sets
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    # Build metadata
    metadata = _build_metadata(train_samples, val_samples, n_samples, val_ratio, seed)

    # Print distribution summaries
    _print_distribution("TRAIN SET", train_samples)
    _print_distribution("VALIDATION SET", val_samples)

    return train_samples, val_samples, metadata


def _build_metadata(
    train_samples: List[OptimizationSample],
    val_samples: List[OptimizationSample],
    n_samples: int,
    val_ratio: float,
    seed: int
) -> Dict[str, Any]:
    """Build metadata dictionary with distribution statistics."""

    def get_distribution(samples: List[OptimizationSample]) -> Dict[str, Any]:
        coll_dist = defaultdict(int)
        ans_dist = defaultdict(int)
        qt_dist = defaultdict(int)

        for s in samples:
            coll_dist[s.collection] += 1
            ans_dist[s.answerability] += 1
            for qt in s.question_types:
                qt_dist[qt] += 1

        return {
            "total": len(samples),
            "by_collection": dict(coll_dist),
            "by_answerability": dict(ans_dist),
            "by_question_type": dict(qt_dist)
        }

    return {
        "created_at": datetime.now().isoformat(),
        "config": {
            "n_samples": n_samples,
            "val_ratio": val_ratio,
            "seed": seed
        },
        "train": get_distribution(train_samples),
        "val": get_distribution(val_samples)
    }


def _print_distribution(label: str, samples: List[OptimizationSample]):
    """Print distribution summary for a set of samples."""
    print(f"\n{'='*60}")
    print(f"{label} DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total: {len(samples)} samples")

    # By Collection
    coll_dist = defaultdict(int)
    for s in samples:
        coll_dist[s.collection] += 1
    print(f"\nBy Collection:")
    for coll, count in sorted(coll_dist.items()):
        print(f"  {coll}: {count}")

    # By Answerability
    ans_dist = defaultdict(int)
    for s in samples:
        ans_dist[s.answerability] += 1
    print(f"\nBy Answerability:")
    for label, count in sorted(ans_dist.items()):
        print(f"  {label}: {count}")

    # By Question Type
    qt_dist = defaultdict(int)
    for s in samples:
        for qt in s.question_types:
            qt_dist[qt] += 1
    print(f"\nBy Question Type:")
    for qt, count in sorted(qt_dist.items()):
        print(f"  {qt}: {count}")


# =============================================================================
# SAVE DATASETS
# =============================================================================

def save_datasets(
    train_samples: List[OptimizationSample],
    val_samples: List[OptimizationSample],
    metadata: Dict[str, Any],
    output_dir: str
):
    """Save train, val datasets and metadata to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save train set
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
    print(f"\nSaved training set to: {train_path}")

    # Save val set
    val_path = os.path.join(output_dir, "val.jsonl")
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved validation set to: {val_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced optimization datasets (train/val) for all pipelines"
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Path to reference JSONL file with labeled data"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Total number of samples to select (default: 100)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help="Ratio of samples for validation (default: 0.2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_data",
        help="Output directory for datasets (default: optimization_data/)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("="*60)
    print("BUILD OPTIMIZATION DATASET")
    print("="*60)
    print(f"Reference file: {args.reference_file}")
    print(f"Total samples: {args.samples}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")

    # Load data
    print("\n[1/3] Loading reference data...")
    data = load_reference_data(args.reference_file)
    print(f"Loaded {len(data)} entries with Answerability labels")

    # Create stratified split
    print("\n[2/3] Creating stratified train/val split...")
    train_samples, val_samples, metadata = stratified_sample_with_split(
        data,
        n_samples=args.samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Save datasets
    print("\n[3/3] Saving datasets...")
    save_datasets(train_samples, val_samples, metadata, args.output_dir)

    print("\n" + "="*60)
    print("DATASET BUILD COMPLETE")
    print("="*60)
    print(f"\nTo use these datasets in optimization pipelines:")
    print(f"  --train-file {args.output_dir}/train.jsonl")
    print(f"  --val-file {args.output_dir}/val.jsonl")


if __name__ == "__main__":
    main()
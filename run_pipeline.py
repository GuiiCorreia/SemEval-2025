#!/usr/bin/env python3
"""
Main Script for Running the Multi-Turn RAG Pipeline

This script provides a complete interface for running retrieval and generation
experiments using the pipeline.

Supports four modes:
- Original pipeline (default)
- DSPy multi-hop RAG (--dspy-multihop) - retrieval only
- DSPy full pipeline (--dspy-full) - retrieval + classification + generation
- Integrated multi-hop (--multihop-retrieval) - DSPy multi-hop retrieval + original generation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import create_pipeline
from src.config import load_config
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval import HybridRetriever
from src.dspy_multihop import create_multihop_retriever
from src.dspy_pipeline import create_integrated_pipeline

# Load environment variables
load_dotenv()

# Collection name mapping (short name -> full name)
COLLECTION_MAPPING = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "ibmcloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}


def resolve_collection_name(name: str) -> str:
    """Resolve collection name from short name or return as-is if already full name."""
    if not name:
        return name
    # Check if it's a short name
    lower_name = name.lower()
    if lower_name in COLLECTION_MAPPING:
        return COLLECTION_MAPPING[lower_name]
    # Return as-is (already full name or unknown)
    return name


def load_queries(queries_file: str) -> List[Dict[str, Any]]:
    """Load queries from JSONL file"""
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries




def run_retrieval_experiment(
    pipeline,
    queries: List[Dict[str, Any]],
    default_collection: str,
    output_file: str
):
    """Run retrieval experiment and save results.

    Each query can have its own collection via the 'Collection' field.
    Falls back to default_collection if not specified.
    """
    print(f"Running retrieval experiment on {len(queries)} queries...")

    results = []
    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}...")

        try:
            # Get collection for this query (use query's collection or fall back to default)
            raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
            collection_name = resolve_collection_name(raw_collection)

            # Extract conversation history from query item
            conversation_history = []
            if 'input' in query_item:
                conversation_history = query_item['input']

            # Get current question
            current_question = query_item.get('text', '')
            if not current_question and conversation_history:
                # If no 'text' field, use the last user message
                for msg in reversed(conversation_history):
                    if msg.get('speaker') == 'user':
                        current_question = msg.get('text', '')
                        break

            if not current_question:
                print(f"Warning: No question found for query {i}")
                continue

            # Process through pipeline (retrieval-only mode)
            response = pipeline.process_query(
                current_question=current_question,
                conversation_history=conversation_history,
                collection_name=collection_name,
                retrieval_only=True
            )

            # Format for Task A retrieval evaluation
            retrieved_docs = []
            # Get detailed retrieval results from pipeline metadata (10 docs)
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])

            # Format contexts for evaluation (document_id, score, text required)
            for doc in retrieval_results[:10]:  # Ensure max 10 docs for Task A
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', '')
                })

            # Task A format: conversation_id, task_id, Collection, input, contexts
            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': retrieved_docs
            }

            # Required fields for Task A
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']

            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")

            # Fallback: save record with empty contexts instead of skipping
            raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
            collection_name = resolve_collection_name(raw_collection) or 'unknown'
            conversation_history = query_item.get('input', [])

            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': []
            }
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            results.append(result)

            print(f"  -> Fallback record saved for query {i}")
            continue

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Retrieval experiment complete! Results saved to {output_file}")


def run_generation_experiment(
    pipeline,
    queries: List[Dict[str, Any]],
    default_collection: str,
    output_file: str
):
    """Run generation experiment and save results.

    Each query can have its own collection via the 'Collection' field.
    Falls back to default_collection if not specified.
    """
    print(f"Running generation experiment on {len(queries)} queries...")

    results = []
    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}...")

        try:
            # Get collection for this query (use query's collection or fall back to default)
            raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
            collection_name = resolve_collection_name(raw_collection)

            # Extract conversation history
            conversation_history = []
            if 'input' in query_item:
                conversation_history = query_item['input']

            # Get current question
            current_question = query_item.get('text', '')
            if not current_question and conversation_history:
                for msg in reversed(conversation_history):
                    if msg.get('speaker') == 'user':
                        current_question = msg.get('text', '')
                        break

            if not current_question:
                print(f"Warning: No question found for query {i}")
                continue

            # Process through pipeline
            response = pipeline.process_query(
                current_question=current_question,
                conversation_history=conversation_history,
                collection_name=collection_name
            )

            # Format for Task C RAG evaluation
            retrieved_docs = []
            # Get detailed retrieval results from pipeline metadata
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])

            # Use only top 5 docs for Task C contexts
            for doc in retrieval_results[:5]:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', '')
                })

            # Task C format: conversation_id, task_id, Collection, input, contexts, predictions
            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': retrieved_docs,
                'predictions': [{
                    'text': response['response_text']
                }]
            }

            # Required fields for Task C
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']

            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")

            # Fallback: try retrieval-only, then save with fallback response
            fallback_response = 'I apologize, but I was unable to process your request due to a technical error.'
            retrieval_results = []

            try:
                # Try retrieval-only as fallback
                response = pipeline.process_query(
                    current_question=current_question,
                    conversation_history=conversation_history,
                    collection_name=collection_name,
                    retrieval_only=True
                )
                retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])
                print(f"  -> Retrieval fallback succeeded: {len(retrieval_results)} docs")
            except Exception as e2:
                print(f"  -> Retrieval fallback also failed: {e2}")

            # Build result with whatever we have
            retrieved_docs = []
            for doc in retrieval_results[:5]:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', '')
                })

            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': retrieved_docs,
                'predictions': [{'text': fallback_response}]
            }
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            results.append(result)

            print(f"  -> Fallback record saved for query {i}")
            continue

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Generation experiment complete! Results saved to {output_file}")


def run_multihop_full_experiment(
    pipeline,
    queries: List[Dict[str, Any]],
    default_collection: str,
    output_retrieval: str,
    output_generation: str
):
    """Run full pipeline once and save both Task A (retrieval) and Task C (RAG) outputs.

    This is more efficient than running retrieval and generation separately
    since it processes each query only once.

    Error handling strategy:
    - If retrieval succeeds but generation fails, keep retrieved docs and use fallback response
    - If retrieval fails completely, use empty contexts and fallback response
    - Never skip a record - always save something
    """
    print(f"Running full pipeline experiment on {len(queries)} queries...")
    print(f"  Task A output (10 docs): {output_retrieval}")
    print(f"  Task C output (5 docs + predictions): {output_generation}")

    results_retrieval = []  # Task A: 10 docs
    results_generation = []  # Task C: 5 docs + predictions
    fallback_response = 'I apologize, but I was unable to process your request due to a technical error.'

    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}...")

        # Extract common fields (safe, no API calls)
        raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
        collection_name = resolve_collection_name(raw_collection) or 'unknown'
        conversation_history = query_item.get('input', [])

        current_question = query_item.get('text', '')
        if not current_question and conversation_history:
            for msg in reversed(conversation_history):
                if msg.get('speaker') == 'user':
                    current_question = msg.get('text', '')
                    break

        # Initialize results with defaults
        retrieval_results = []
        response_text = fallback_response
        retrieval_success = False

        # Skip if no question found
        if not current_question:
            print(f"Warning: No question found for query {i}, using fallback")
        else:
            # Try to run the full pipeline
            try:
                response = pipeline.process_query(
                    current_question=current_question,
                    conversation_history=conversation_history,
                    collection_name=collection_name,
                    retrieval_only=False  # Full pipeline
                )

                # Extract retrieval results (even if generation failed later)
                retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])
                retrieval_success = len(retrieval_results) > 0

                # Extract response text (may be empty if generation failed)
                response_text = response.get('response_text', '') or fallback_response

            except Exception as e:
                print(f"Error in full pipeline for query {i}: {e}")
                import traceback
                traceback.print_exc()

                # Try retrieval-only as fallback
                try:
                    print(f"  -> Attempting retrieval-only fallback...")
                    response = pipeline.process_query(
                        current_question=current_question,
                        conversation_history=conversation_history,
                        collection_name=collection_name,
                        retrieval_only=True  # Only retrieval
                    )
                    retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])
                    retrieval_success = len(retrieval_results) > 0
                    if retrieval_success:
                        print(f"  -> Retrieval fallback succeeded: {len(retrieval_results)} docs")
                except Exception as e2:
                    print(f"  -> Retrieval fallback also failed: {e2}")

        # === Task A: Retrieval (10 docs) ===
        docs_task_a = []
        for doc in retrieval_results[:10]:
            docs_task_a.append({
                'document_id': doc.get('document_id', ''),
                'score': doc.get('final_score', doc.get('score', 1.0)),
                'text': doc.get('text', ''),
                'title': doc.get('title', ''),
                'source': doc.get('source', '')
            })

        result_a = {
            'Collection': collection_name,
            'input': conversation_history,
            'contexts': docs_task_a
        }
        if 'conversation_id' in query_item:
            result_a['conversation_id'] = query_item['conversation_id']
        if 'task_id' in query_item:
            result_a['task_id'] = query_item['task_id']
        results_retrieval.append(result_a)

        # === Task C: RAG (5 docs + predictions) ===
        docs_task_c = []
        for doc in retrieval_results[:5]:
            docs_task_c.append({
                'document_id': doc.get('document_id', ''),
                'score': doc.get('final_score', doc.get('score', 1.0)),
                'text': doc.get('text', ''),
                'title': doc.get('title', ''),
                'source': doc.get('source', '')
            })

        result_c = {
            'Collection': collection_name,
            'input': conversation_history,
            'contexts': docs_task_c,
            'predictions': [{
                'text': response_text
            }]
        }
        if 'conversation_id' in query_item:
            result_c['conversation_id'] = query_item['conversation_id']
        if 'task_id' in query_item:
            result_c['task_id'] = query_item['task_id']
        results_generation.append(result_c)

        # Log if fallback was used
        if not retrieval_success or response_text == fallback_response:
            print(f"  -> Query {i}: retrieval={'OK' if retrieval_success else 'FALLBACK'}, generation={'OK' if response_text != fallback_response else 'FALLBACK'}")

    # Save Task A results
    print(f"\nSaving {len(results_retrieval)} Task A results to {output_retrieval}...")
    output_path = Path(output_retrieval)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_retrieval, 'w', encoding='utf-8') as f:
        for result in results_retrieval:
            f.write(json.dumps(result) + '\n')

    # Save Task C results
    output_path = Path(output_generation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(results_generation)} Task C results to {output_generation}...")
    with open(output_generation, 'w', encoding='utf-8') as f:
        for result in results_generation:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Full pipeline experiment complete!")
    print(f"  Task A: {output_retrieval}")
    print(f"  Task C: {output_generation}")


def run_dspy_full_experiment(
    pipeline,
    queries: List[Dict[str, Any]],
    default_collection: str,
    output_file: str
):
    """Run DSPy full pipeline experiment (retrieval + classification + generation).

    Each query can have its own collection via the 'Collection' field.
    Falls back to default_collection if not specified.
    """
    print(f"Running DSPy full pipeline experiment on {len(queries)} queries...")

    results = []
    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"\nProcessing query {i+1}/{len(queries)}...")

        try:
            # Get collection for this query
            raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
            collection_name = resolve_collection_name(raw_collection)

            # Extract conversation history
            conversation_history = []
            if 'input' in query_item:
                conversation_history = query_item['input']

            # Get current question
            current_question = query_item.get('text', '')
            if not current_question and conversation_history:
                for msg in reversed(conversation_history):
                    if msg.get('speaker') == 'user':
                        current_question = msg.get('text', '')
                        break

            if not current_question:
                print(f"Warning: No question found for query {i}")
                continue

            # Process through full pipeline
            response = pipeline.process_query(
                current_question=current_question,
                conversation_history=conversation_history,
                collection_name=collection_name
            )

            # Format for Task C RAG evaluation
            retrieved_docs = []
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])

            # Use only top 5 docs for Task C contexts
            for doc in retrieval_results[:5]:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', '')
                })

            # Task C format: conversation_id, task_id, Collection, input, contexts, predictions
            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': retrieved_docs,
                'predictions': [{
                    'text': response.get('response_text', '')
                }]
            }

            # Required fields for Task C
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']

            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: save record with empty contexts and default response
            raw_collection = query_item.get('Collection') or query_item.get('collection') or default_collection
            collection_name = resolve_collection_name(raw_collection) or 'unknown'
            conversation_history = query_item.get('input', [])

            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'contexts': [],
                'predictions': [{
                    'text': 'I apologize, but I was unable to process your request due to a technical error.'
                }]
            }
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            results.append(result)

            print(f"  -> Fallback record saved for query {i}")
            continue

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ DSPy full pipeline experiment complete! Results saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run Multi-Turn RAG Pipeline Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mode',
        choices=['retrieval', 'generation', 'both'],
        default='both',
        help='Experiment mode'
    )
    parser.add_argument(
        '--queries',
        type=str,
        required=True,
        help='Path to queries JSONL file'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Collection name to search (auto-detected from queries if not provided)'
    )
    parser.add_argument(
        '--output-retrieval',
        type=str,
        default='retrieval_results.jsonl',
        help='Output file for retrieval results'
    )
    parser.add_argument(
        '--output-generation',
        type=str,
        default='generation_results.jsonl',
        help='Output file for generation results'
    )
    parser.add_argument(
        '--index-corpus',
        type=str,
        help='Path to corpus file to index before running experiments'
    )
    parser.add_argument(
        '--retrieval-mode',
        type=str,
        choices=['hybrid', 'dense_only', 'bm25_only'],
        default=None,
        help='Retrieval mode: hybrid (both), dense_only, or bm25_only. If not specified, uses config/env default (hybrid)'
    )
    parser.add_argument(
        '--dspy-multihop',
        action='store_true',
        help='Use DSPy multi-hop RAG instead of the original pipeline'
    )
    parser.add_argument(
        '--num-hops',
        type=int,
        default=3,
        help='Number of retrieval hops for DSPy multi-hop mode'
    )
    parser.add_argument(
        '--dspy-model',
        type=str,
        default='openai/openai/gpt-oss-120b',
        help='LLM model to use for DSPy modes (e.g., openai/gpt-oss-120b, gemini/gemini-3-flash-preview)'
    )
    parser.add_argument(
        '--dspy-base-url',
        type=str,
        default='https://api.deepinfra.com/v1/openai',
        help='Base URL for OpenAI-compatible API (e.g., DeepInfra, Together, etc.)'
    )
    parser.add_argument(
        '--dspy-full',
        action='store_true',
        help='Use DSPy full pipeline (retrieval + classification + generation)'
    )
    parser.add_argument(
        '--retriever-path',
        type=str,
        default=None,
        help='Path to optimized retriever weights (for --dspy-full or --dspy-multihop)'
    )
    parser.add_argument(
        '--classifier-path',
        type=str,
        default=None,
        help='Path to optimized classifier weights (for --dspy-full)'
    )
    parser.add_argument(
        '--generator-path',
        type=str,
        default=None,
        help='Path to optimized generator weights (for --dspy-full)'
    )
    parser.add_argument(
        '--multihop-retrieval',
        action='store_true',
        help='Use integrated mode: DSPy multi-hop retrieval + original answerability/generation'
    )

    args = parser.parse_args()

    # Resolve collection name if provided (supports short names like 'fiqa', 'govt', etc.)
    if args.collection:
        args.collection = resolve_collection_name(args.collection)

    # Load queries first
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries")

    # =========================================================================
    # INTEGRATED MODE: DSPy Multi-hop Retrieval + Original Generation
    # =========================================================================
    if args.multihop_retrieval:
        print("\n" + "="*60)
        print("INTEGRATED MODE: DSPy Multi-hop Retrieval + Original Generation")
        print("="*60)
        print(f"  Model: {args.dspy_model}")
        print(f"  Base URL: {args.dspy_base_url}")
        print(f"  Hops: {args.num_hops}")
        if args.retriever_path:
            print(f"  Optimized retriever: {args.retriever_path}")

        try:
            import dspy

            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            # Configure DSPy with ChatAdapter (disable JSON fallback for compatibility)
            lm = dspy.LM(
                args.dspy_model,
                api_key=api_key,
                base_url=args.dspy_base_url,
                temperature=0.4,
                max_tokens=10000
            )
            dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

            # Create integrated pipeline with multi-hop retrieval
            pipeline = create_pipeline(
                use_multihop=True,
                multihop_model_path=args.retriever_path,
                num_hops=args.num_hops
            )

            # Override retrieval mode if specified
            if args.retrieval_mode:
                print(f"Overriding retrieval mode to: {args.retrieval_mode}")
                pipeline.hybrid_retriever.config.retrieval_mode = args.retrieval_mode

            # Index corpus if provided
            if args.index_corpus:
                print(f"Indexing corpus {args.index_corpus}...")
                collection_for_index = args.collection or "default"
                success = pipeline.setup_collection(collection_for_index, args.index_corpus)
                if not success:
                    print("Failed to index corpus!")
                    return 1

            print("Integrated pipeline initialized!")

            # Run experiments based on mode
            if args.mode == 'both':
                # Single pass: generates both Task A and Task C files
                run_multihop_full_experiment(
                    pipeline, queries, args.collection,
                    args.output_retrieval, args.output_generation
                )
            elif args.mode == 'retrieval':
                run_retrieval_experiment(
                    pipeline, queries, args.collection, args.output_retrieval
                )
            elif args.mode == 'generation':
                run_generation_experiment(
                    pipeline, queries, args.collection, args.output_generation
                )

            print("\n✓ All experiments completed successfully!")
            return 0

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Check if DSPy full pipeline mode is enabled
    if args.dspy_full:
        print("Initializing DSPy Full Pipeline (Retrieval + Classification + Generation)...")
        print(f"  Model: {args.dspy_model}")
        print(f"  Base URL: {args.dspy_base_url}")
        print(f"  Hops: {args.num_hops}")

        try:
            import dspy

            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            # Configure DSPy with retrieval model
            lm = dspy.LM(args.dspy_model, api_key=api_key, base_url=args.dspy_base_url, temperature=0.4, max_tokens=10000)
            dspy.configure(lm=lm)

            # Initialize retrieval infrastructure
            config = load_config()
            embedding_service = EmbeddingService(config.embedding)
            vector_store = VectorStore(config.qdrant, embedding_service)
            hybrid_retriever = HybridRetriever(config.retrieval, vector_store)

            # Override retrieval mode if specified
            if args.retrieval_mode:
                print(f"Overriding retrieval mode to: {args.retrieval_mode}")
                hybrid_retriever.config.retrieval_mode = args.retrieval_mode

            # Create integrated pipeline
            pipeline = create_integrated_pipeline(
                hybrid_retriever=hybrid_retriever,
                collection_name=args.collection,
                num_hops=args.num_hops,
                api_key=api_key,
                retriever_path=args.retriever_path,
                classifier_path=args.classifier_path,
                generator_path=args.generator_path
            )

            print("DSPy Full Pipeline initialized!")

            # Run full pipeline experiment
            run_dspy_full_experiment(
                pipeline, queries, args.collection, args.output_generation
            )

            print("\n✓ All experiments completed successfully!")
            return 0

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Check if DSPy multi-hop mode is enabled
    if args.dspy_multihop:
        print("Initializing DSPy Multi-Hop RAG Pipeline...")
        print(f"  Model: {args.dspy_model}")
        print(f"  Base URL: {args.dspy_base_url}")
        print(f"  Hops: {args.num_hops}")

        try:
            # Configure DSPy
            import dspy

            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            lm = dspy.LM(args.dspy_model, api_key=api_key, base_url=args.dspy_base_url, temperature=0.4, max_tokens=10000)
            dspy.configure(lm=lm)

            # Initialize retrieval infrastructure
            config = load_config()
            embedding_service = EmbeddingService(config.embedding)
            vector_store = VectorStore(config.qdrant, embedding_service)
            hybrid_retriever = HybridRetriever(config.retrieval, vector_store)

            # Override retrieval mode if specified
            if args.retrieval_mode:
                print(f"Overriding retrieval mode to: {args.retrieval_mode}")
                hybrid_retriever.config.retrieval_mode = args.retrieval_mode

            # Index corpus if provided
            if args.index_corpus:
                print(f"Indexing corpus {args.index_corpus}...")
                collection_for_index = args.collection or "default"
                bm25_success = hybrid_retriever.index_corpus_bm25(collection_for_index, args.index_corpus)
                if not bm25_success:
                    print("Failed to index corpus for BM25!")
                    return 1

            # Create DSPy multi-hop retriever (collection will be taken from each query)
            pipeline = create_multihop_retriever(
                hybrid_retriever=hybrid_retriever,
                collection_name=args.collection,  # Default, can be overridden per-query
                num_hops=args.num_hops
            )

            print("DSPy Multi-Hop RAG Pipeline initialized!")

            # Run experiments (only retrieval mode supported for DSPy multi-hop)
            if args.mode in ['retrieval', 'both']:
                run_retrieval_experiment(
                    pipeline, queries, args.collection, args.output_retrieval
                )

            if args.mode in ['generation', 'both']:
                print("Warning: Generation mode not yet supported with DSPy multi-hop. Skipping.")

            print("\n✓ All experiments completed successfully!")
            return 0

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Original pipeline mode
    print("Initializing Multi-Turn RAG Pipeline...")

    try:
        # Create pipeline
        pipeline = create_pipeline()

        # Override retrieval mode if specified via argument
        if args.retrieval_mode:
            print(f"Overriding retrieval mode to: {args.retrieval_mode}")
            pipeline.hybrid_retriever.config.retrieval_mode = args.retrieval_mode

        # Index corpus if provided
        if args.index_corpus:
            collection_for_index = args.collection or "default"
            print(f"Indexing corpus {args.index_corpus}...")
            success = pipeline.setup_collection(collection_for_index, args.index_corpus)
            if not success:
                print("Failed to index corpus!")
                return 1

        # Run experiments (queries already loaded above)
        if args.mode in ['retrieval', 'both']:
            run_retrieval_experiment(
                pipeline, queries, args.collection, args.output_retrieval
            )

        if args.mode in ['generation', 'both']:
            run_generation_experiment(
                pipeline, queries, args.collection, args.output_generation
            )

        print("\n✓ All experiments completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
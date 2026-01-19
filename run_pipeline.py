#!/usr/bin/env python3
"""
Main Script for Running the Multi-Turn RAG Pipeline

This script provides a complete interface for running retrieval and generation
experiments using the pipeline.

Supports three modes:
- Original pipeline (default)
- DSPy multi-hop RAG (--dspy-multihop) - retrieval only
- DSPy full pipeline (--dspy-full) - retrieval + classification + generation
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
            collection_name = query_item.get('Collection', default_collection)

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

            # Format for retrieval evaluation (compatible with evaluation template)
            retrieved_docs = []
            # Get detailed retrieval results from pipeline metadata
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])

            # Use detailed retrieval results (should always be available)
            for doc in retrieval_results:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'source': doc.get('source', ''),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', '')
                })

            result = {
                'task_id': query_item.get('_id', f"query_{i}"),
                'Collection': collection_name,
                'contexts': retrieved_docs,
                'input': conversation_history,
                'targets': query_item.get('targets', [{'speaker': 'agent', 'text': ''}]),
                'metadata': {
                    'reference_contexts': query_item.get('contexts', []),
                    'pipeline_metadata': response.get('pipeline_metadata', {})
                }
            }

            # Propagate original metadata from input
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            if 'task_type' in query_item:
                result['task_type'] = query_item['task_type']
            if 'turn' in query_item:
                result['turn'] = query_item['turn']
            if 'Question Type' in query_item:
                result['Question Type'] = query_item['Question Type']
            if 'No. References' in query_item:
                result['No. References'] = query_item['No. References']
            if 'Multi-Turn' in query_item:
                result['Multi-Turn'] = query_item['Multi-Turn']
            if 'Answerability' in query_item:
                result['Answerability'] = query_item['Answerability']
            if 'dataset' in query_item:
                result['dataset'] = query_item['dataset']

            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue

    # Save results
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
            collection_name = query_item.get('Collection', default_collection)

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

            # Format for generation evaluation (compatible with evaluation template)
            retrieved_docs = []
            # Get detailed retrieval results from pipeline metadata
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])

            # Use detailed retrieval results (should always be available)
            for doc in retrieval_results:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'source': doc.get('source', ''),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', '')
                })

            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'targets': query_item.get('targets', [{
                    'speaker': 'agent',
                    'text': ''
                }]),
                'predictions': [{
                    'text': response['response_text']
                }],
                'contexts': retrieved_docs,
                'metadata': {
                    'confidence_score': response.get('confidence_score', 0.0),
                    'answerability': response.get('pipeline_metadata', {}).get('answerability', 'A'),
                    'guardrail_flags': response.get('guardrail_flags', {}),
                    'pipeline_metadata': response.get('pipeline_metadata', {})
                }
            }

            # Propagate original metadata from input
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            if 'task_type' in query_item:
                result['task_type'] = query_item['task_type']
            if 'turn' in query_item:
                result['turn'] = query_item['turn']
            if 'Question Type' in query_item:
                result['Question Type'] = query_item['Question Type']
            if 'No. References' in query_item:
                result['No. References'] = query_item['No. References']
            if 'Multi-Turn' in query_item:
                result['Multi-Turn'] = query_item['Multi-Turn']
            if 'Answerability' in query_item:
                result['Answerability'] = query_item['Answerability']
            if 'dataset' in query_item:
                result['dataset'] = query_item['dataset']
            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue

    # Save results
    print(f"Saving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Generation experiment complete! Results saved to {output_file}")


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
            collection_name = query_item.get('Collection', default_collection)

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

            # Format retrieved documents
            retrieved_docs = []
            retrieval_results = response.get('pipeline_metadata', {}).get('retrieved_documents', [])
            for doc in retrieval_results:
                retrieved_docs.append({
                    'document_id': doc.get('document_id', ''),
                    'score': doc.get('final_score', doc.get('score', 1.0)),
                    'source': doc.get('source', ''),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', '')
                })

            # Build result in evaluation-compatible format
            result = {
                'Collection': collection_name,
                'input': conversation_history,
                'targets': query_item.get('targets', [{
                    'speaker': 'agent',
                    'text': ''
                }]),
                'predictions': [{
                    'text': response.get('response_text', '')
                }],
                'contexts': retrieved_docs,
                'metadata': {
                    'classification': response.get('classification', ''),
                    'pipeline_metadata': response.get('pipeline_metadata', {})
                }
            }

            # Propagate original metadata from input
            if 'conversation_id' in query_item:
                result['conversation_id'] = query_item['conversation_id']
            if 'task_id' in query_item:
                result['task_id'] = query_item['task_id']
            if 'task_type' in query_item:
                result['task_type'] = query_item['task_type']
            if 'turn' in query_item:
                result['turn'] = query_item['turn']
            if 'Question Type' in query_item:
                result['Question Type'] = query_item['Question Type']
            if 'No. References' in query_item:
                result['No. References'] = query_item['No. References']
            if 'Multi-Turn' in query_item:
                result['Multi-Turn'] = query_item['Multi-Turn']
            if 'Answerability' in query_item:
                result['Answerability'] = query_item['Answerability']
            if 'dataset' in query_item:
                result['dataset'] = query_item['dataset']

            results.append(result)

        except Exception as e:
            print(f"Error processing query {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
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
        default='gemini/gemini-3-flash-preview',
        help='LLM model to use for DSPy multi-hop (e.g., gemini/gemini-2.5-flash-preview, groq/llama-3.3-70b-versatile)'
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

    args = parser.parse_args()

    # Load queries first
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries")

    # Check if DSPy full pipeline mode is enabled
    if args.dspy_full:
        print("Initializing DSPy Full Pipeline (Retrieval + Classification + Generation)...")
        print(f"  Retrieval model: {args.dspy_model}")
        print(f"  Hops: {args.num_hops}")
        print(f"  Classification: gemini/gemini-3-flash-preview")
        print(f"  Generation: gemini/gemini-3-pro-preview")

        try:
            import dspy

            # Get API key
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            # Configure DSPy with retrieval model
            lm = dspy.LM(args.dspy_model, api_key=api_key, temperature=0.4, max_tokens=8000)
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
        print(f"  Hops: {args.num_hops}")

        try:
            # Configure DSPy
            import dspy

            # Get API key based on model provider
            if args.dspy_model.startswith('gemini/'):
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment")
            elif args.dspy_model.startswith('groq/'):
                api_key = os.getenv('GROQ_API_KEY')
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment")
            else:
                api_key = os.getenv('OPENAI_API_KEY')

            lm = dspy.LM(args.dspy_model, api_key=api_key, temperature=0.4, max_tokens=8000)
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
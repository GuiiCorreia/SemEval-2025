#!/usr/bin/env python3
"""
Main Script for Running the Multi-Turn RAG Pipeline

This script provides a complete interface for running retrieval and generation
experiments using the pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import create_pipeline

# Load environment variables
load_dotenv()

# =============================================================================
# COLLECTION NAME MAPPING
# =============================================================================
# Maps short collection names to full Qdrant collection names (BM25 index names)
COLLECTION_MAPPING = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "ibmcloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

def resolve_collection_name(collection: str) -> str:
    """Resolve short collection name to full Qdrant collection name."""
    return COLLECTION_MAPPING.get(collection.lower(), collection)


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
    collection_name: str,
    output_file: str
):
    """Run retrieval experiment and save results"""
    print(f"Running retrieval experiment on {len(queries)} queries...")

    results = []
    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}...")

        try:
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
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Retrieval experiment complete! Results saved to {output_file}")


def run_generation_experiment(
    pipeline,
    queries: List[Dict[str, Any]],
    collection_name: str,
    output_file: str
):
    """Run generation experiment and save results"""
    print(f"Running generation experiment on {len(queries)} queries...")
    
    results = []
    for i, query_item in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}...")
        
        try:
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
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Generation experiment complete! Results saved to {output_file}")


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
        required=True,
        help='Collection name to search'
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

    args = parser.parse_args()
    
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
            print(f"Indexing corpus {args.index_corpus}...")
            success = pipeline.setup_collection(args.collection, args.index_corpus)
            if not success:
                print("Failed to index corpus!")
                return 1
        
        # Load queries
        print(f"Loading queries from {args.queries}...")
        queries = load_queries(args.queries)
        print(f"Loaded {len(queries)} queries")

        # Resolve collection name
        collection_name = resolve_collection_name(args.collection)
        if collection_name != args.collection:
            print(f"Resolved collection: {args.collection} -> {collection_name}")

        # Run experiments
        if args.mode in ['retrieval', 'both']:
            run_retrieval_experiment(
                pipeline, queries, collection_name, args.output_retrieval
            )

        if args.mode in ['generation', 'both']:
            run_generation_experiment(
                pipeline, queries, collection_name, args.output_generation
            )
        
        print("\n✓ All experiments completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
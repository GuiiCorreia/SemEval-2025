#!/usr/bin/env python3
"""
Corpus Indexing Script

This script indexes a corpus for both dense (Qdrant) and sparse (BM25) retrieval.
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_pipeline, load_config

# Load environment variables
load_dotenv()


def main():
    """Main indexing function"""
    parser = argparse.ArgumentParser(
        description='Index a corpus for hybrid retrieval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--corpus', 
        type=str, 
        required=True, 
        help='Path to corpus JSONL file'
    )
    parser.add_argument(
        '--collection', 
        type=str, 
        required=True, 
        help='Name for the collection (e.g., clapnq)'
    )
    parser.add_argument(
        '--check-env', 
        action='store_true', 
        help='Check environment variables before indexing'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for embedding documents'
    )
    parser.add_argument(
        '--bm25-only',
        action='store_true',
        help='Index only BM25 (skip dense vector indexing)'
    )

    args = parser.parse_args()
    
    # Check if corpus file exists
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file {args.corpus} not found!")
        return 1
    
    # Check environment variables if requested
    if args.check_env:
        print("Checking environment variables...")
        required_vars = ['GEMINI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"Error: Missing environment variables: {missing_vars}")
            print("Please set these variables in your .env file")
            return 1
        print("Environment variables OK!")
    
    print(f"Starting indexing process...")
    print(f"Corpus file: {args.corpus}")
    print(f"Collection name: {args.collection}")
    
    try:
        # Create pipeline
        print("\nInitializing pipeline...")
        if args.config:
            print(f"Using configuration file: {args.config}")
        pipeline = create_pipeline(args.config)

        # Override batch size if specified
        if args.batch_size:
            pipeline.vector_store.embedding_service.config.batch_size = args.batch_size
            print(f"Using batch size: {args.batch_size}")
        
        # Setup collection (indexes both vector and BM25, or BM25-only)
        print(f"\nIndexing collection {args.collection}...")
        if args.bm25_only:
            print("BM25-only mode enabled")
        success = pipeline.setup_collection(args.collection, args.corpus, bm25_only=args.bm25_only)
        
        if success:
            print(f"\n✓ Successfully indexed collection {args.collection}!")
            
            # Show collection info
            print("\nCollection information:")
            info = pipeline.get_collection_info(args.collection)
            if info:
                print(f"  - Vectors count: {info.get('vectors_count', 'N/A')}")
                print(f"  - Status: {info.get('status', 'N/A')}")
            
            return 0
        else:
            print(f"\n✗ Failed to index collection {args.collection}")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during indexing: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
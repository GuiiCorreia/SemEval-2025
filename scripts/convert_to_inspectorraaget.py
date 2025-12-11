#!/usr/bin/env python3
"""
Script to convert retrieval evaluation results to InspectorRAGet format
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime

def load_jsonl(file_path):
    """Load JSONL file"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_documents(data):
    """Extract unique documents from all contexts"""
    documents = {}
    for item in data:
        for context in item.get('contexts', []):
            doc_id = context['document_id']
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "text": context['text'],
                    "title": context.get('title', ''),
                }
                if context.get('reference'):
                    documents[doc_id]["reference"] = True
    return list(documents.values())

def create_tasks(data):
    """Convert data to InspectorRAGet tasks format"""
    tasks = []
    
    for item in data:
        task = {
            "task_id": item['task_id'],
            "domain": item.get('Collection', ''),
            "question": item['input'][-1]['text'],  # Last user message
            "answer": item['targets'][0]['text'] if item['targets'] else '',
            "contexts": [
                {
                    "document_id": ctx['document_id'],
                    "score": ctx['score']
                }
                for ctx in item.get('contexts', [])
            ],
            "metadata": {
                "conversation_id": item.get('conversation_id', ''),
                "turn": item.get('turn', 1),
                "question_type": item.get('Question Type', []),
                "answerability": item.get('Answerability', []),
                "multi_turn": item.get('Multi-Turn', []),
                "collection": item.get('Collection', ''),
                "rewritten_query": item.get('rewritten_query'),
            }
        }
        
        # Add retrieval metrics if available
        if 'retriever_scores' in item:
            task["retrieval_metrics"] = item['retriever_scores']
        
        tasks.append(task)
    
    return tasks

def calculate_aggregate_metrics(data):
    """Calculate aggregate metrics across all tasks"""
    metrics = defaultdict(list)
    
    for item in data:
        if 'retriever_scores' in item and item['retriever_scores']:
            for metric, value in item['retriever_scores'].items():
                if isinstance(value, (int, float)):
                    metrics[metric].append(value)
    
    # Calculate averages
    aggregate_metrics = {}
    for metric, values in metrics.items():
        if values:
            aggregate_metrics[metric] = {
                "mean": sum(values) / len(values),
                "count": len(values),
                "min": min(values),
                "max": max(values)
            }
    
    return aggregate_metrics

def convert_to_inspectorraaget(input_file, output_file, dataset_name=None):
    """Convert retrieval results to InspectorRAGet format"""
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    
    if not data:
        print("No data found!")
        return
    
    print(f"Found {len(data)} tasks")
    
    # Extract unique collections/domains
    collections = list(set(item.get('Collection', 'Unknown') for item in data))
    
    # Get dataset name
    if not dataset_name:
        dataset_name = f"Retrieval Results ({collections[0] if collections else 'Unknown'})"
    
    # Create InspectorRAGet format
    inspector_format = {
        "name": dataset_name,
        "description": f"Retrieval evaluation results converted on {datetime.now().isoformat()}",
        "domains": collections,
        "models": [
            {
                "model_id": "retriever_system",
                "name": "Retrieval System", 
                "owner": "Evaluation"
            }
        ],
        "metrics": [
            {
                "name": "recall_1",
                "display_name": "Recall@1",
                "description": "Recall at rank 1",
                "author": "algorithm",
                "type": "numerical",
                "aggregator": "average",
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            },
            {
                "name": "recall_3", 
                "display_name": "Recall@3",
                "description": "Recall at rank 3",
                "author": "algorithm",
                "type": "numerical", 
                "aggregator": "average",
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            },
            {
                "name": "recall_5",
                "display_name": "Recall@5", 
                "description": "Recall at rank 5",
                "author": "algorithm",
                "type": "numerical",
                "aggregator": "average", 
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            },
            {
                "name": "ndcg_cut_1",
                "display_name": "nDCG@1",
                "description": "Normalized Discounted Cumulative Gain at rank 1", 
                "author": "algorithm",
                "type": "numerical",
                "aggregator": "average",
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            },
            {
                "name": "ndcg_cut_3",
                "display_name": "nDCG@3",
                "description": "Normalized Discounted Cumulative Gain at rank 3",
                "author": "algorithm", 
                "type": "numerical",
                "aggregator": "average",
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            },
            {
                "name": "ndcg_cut_5",
                "display_name": "nDCG@5",
                "description": "Normalized Discounted Cumulative Gain at rank 5",
                "author": "algorithm",
                "type": "numerical", 
                "aggregator": "average",
                "range": [0, 1, 0.1],
                "min_value": 0,
                "max_value": 1
            }
        ],
        "documents": extract_documents(data),
        "tasks": create_tasks(data),
        "aggregate_metrics": calculate_aggregate_metrics(data)
    }
    
    # Save converted file
    print(f"Saving converted data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(inspector_format, f, indent=2)
    
    print("Conversion completed!")
    print(f"- {len(inspector_format['tasks'])} tasks")
    print(f"- {len(inspector_format['documents'])} documents")
    print(f"- {len(inspector_format['metrics'])} metrics")
    print(f"- Collections: {', '.join(collections)}")

def main():
    parser = argparse.ArgumentParser(description='Convert retrieval results to InspectorRAGet format')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file with retrieval results')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file for InspectorRAGet')
    parser.add_argument('--name', '-n', help='Dataset name (optional)')
    
    args = parser.parse_args()
    
    convert_to_inspectorraaget(args.input, args.output, args.name)

if __name__ == "__main__":
    main()
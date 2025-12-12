"""
Vector Store Module using Qdrant

This module handles vector storage and retrieval using Qdrant vector database.
"""

import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from .config import QdrantConfig
from .embeddings import EmbeddingService

# Load environment variables
load_dotenv()


class VectorStore:
    """Vector store implementation using Qdrant"""
    
    def __init__(self, config: QdrantConfig, embedding_service: EmbeddingService):
        """
        Initialize the vector store
        
        Args:
            config: Qdrant configuration
            embedding_service: Service for generating embeddings
        """
        self.config = config
        self.embedding_service = embedding_service
        self.client = QdrantClient(
            url=config.url,
            api_key=config.api_key
        )
        
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            True if collection was created successfully
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(collection.name == collection_name for collection in collections):
                print(f"Collection {collection_name} already exists")
                return True
                
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_service.config.dimension,
                    distance=getattr(models.Distance, self.config.distance_metric)
                )
            )
            print(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            return False
    
    def _get_checkpoint_file(self, collection_name: str) -> str:
        """Get checkpoint file path for a collection"""
        return f".checkpoint_{collection_name}.json"

    def _load_checkpoint(self, collection_name: str) -> int:
        """Load checkpoint to resume indexing"""
        checkpoint_file = self._get_checkpoint_file(collection_name)
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('last_indexed', 0)
        return 0

    def _save_checkpoint(self, collection_name: str, last_indexed: int):
        """Save checkpoint after successful batch"""
        checkpoint_file = self._get_checkpoint_file(collection_name)
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_indexed': last_indexed}, f)

    def _clear_checkpoint(self, collection_name: str):
        """Clear checkpoint after successful completion"""
        checkpoint_file = self._get_checkpoint_file(collection_name)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    def index_corpus(self, collection_name: str, corpus_file: str, resume: bool = True) -> bool:
        """
        Index a corpus into the vector store with checkpoint/resume support

        Args:
            collection_name: Name of the collection to store documents
            corpus_file: Path to the corpus JSONL file
            resume: Whether to resume from checkpoint if available

        Returns:
            True if indexing was successful
        """
        print(f"Loading corpus from {corpus_file}...")

        # Load documents from JSONL file
        documents = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        total_docs = len(documents)
        print(f"Loaded {total_docs} documents")

        # Create collection if it doesn't exist
        if not self.create_collection(collection_name):
            return False

        # Check for checkpoint
        start_from = 0
        if resume:
            start_from = self._load_checkpoint(collection_name)
            if start_from > 0:
                print(f"Resuming from checkpoint: {start_from} documents already indexed")

        # Index documents in batches
        batch_size = self.embedding_service.config.batch_size

        for i in range(start_from, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_end = min(i + batch_size, total_docs)

            print(f"Indexing documents {i+1}-{batch_end}/{total_docs}...")

            try:
                # Extract texts for batch embedding
                texts = [doc['text'] for doc in batch]

                # Generate embeddings for all documents in batch at once
                embeddings = self.embedding_service.embed_documents(texts)

                # Prepare points for upsert
                points = []
                for idx, (doc, embedding) in enumerate(zip(batch, embeddings)):
                    # Create point with document data as payload
                    point = models.PointStruct(
                        id=i + idx,  # Use global index as ID
                        vector=embedding,
                        payload={
                            '_id': doc['_id'],
                            'text': doc['text'],
                            'title': doc.get('title', ''),
                            'metadata': doc.get('metadata', {})
                        }
                    )
                    points.append(point)

                # Upsert batch to Qdrant
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )

                # Save checkpoint after successful batch
                self._save_checkpoint(collection_name, batch_end)
                print(f"✓ Batch indexed successfully (checkpoint saved)")

            except Exception as e:
                print(f"✗ Error indexing batch {i+1}-{batch_end}: {e}")
                print(f"Progress saved. You can resume from document {i+1} by running the command again.")
                return False

        # Clear checkpoint on successful completion
        self._clear_checkpoint(collection_name)
        print(f"✓ Successfully indexed {total_docs} documents in collection {collection_name}")
        return True
    
    def search(self, collection_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            collection_name: Name of the collection to search
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_query(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                doc = {
                    'document_id': result.payload['_id'],
                    'text': result.payload['text'],
                    'title': result.payload['title'],
                    'score': float(result.score),
                    'metadata': result.payload.get('metadata', {})
                }
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'status': info.status,
                'config': info.config
            }
        except Exception as e:
            print(f"Error getting collection info for {collection_name}: {e}")
            return {}
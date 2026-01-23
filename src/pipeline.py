"""
Main Pipeline Module

This module orchestrates the complete multi-turn RAG pipeline following the 
ERIGO2025 architecture.
"""

from typing import List, Dict, Any, Optional
from .config import SystemConfig, load_config
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .retrieval import HybridRetriever
from .query_reformulation import QueryReformulator, QueryDiversifier
from .answerability import AnswerabilityDetector, AnswerabilityType
from .response_generation import ResponseGenerator
from .dspy_multihop import create_multihop_retriever


class Pipeline:
    """
    Main pipeline class implementing the ERIGO2025 multi-turn RAG architecture
    
    Phases:
    1. Context-Aware Query Reformulation
    2. Multi-Strategy Query Diversification  
    3. Hybrid Document Retrieval
    4. Cross-Encoder Reranking
    5. Answerability Classification
    6. Conditional Response Generation
    """
    
    def __init__(
        self,
        config: SystemConfig = None,
        use_multihop: bool = False,
        multihop_model_path: Optional[str] = None,
        num_hops: int = 3
    ):
        """
        Initialize the pipeline

        Args:
            config: System configuration (loads default if None)
            use_multihop: If True, use DSPy multi-hop retrieval instead of standard retrieval
            multihop_model_path: Path to optimized multi-hop model (e.g., 'optimized_multihop.json')
            num_hops: Number of retrieval hops for multi-hop mode (default: 3)
        """
        self.config = config or load_config()
        self.use_multihop = use_multihop
        self.multihop_model_path = multihop_model_path
        self.num_hops = num_hops

        # Initialize core components
        self.embedding_service = EmbeddingService(self.config.embedding)
        self.vector_store = VectorStore(self.config.qdrant, self.embedding_service)
        self.hybrid_retriever = HybridRetriever(self.config.retrieval, self.vector_store)

        # Initialize retrieval components based on mode
        if use_multihop:
            print("Using DSPy Multi-Hop Retrieval mode")
            self.multihop_retriever = create_multihop_retriever(
                hybrid_retriever=self.hybrid_retriever,
                collection_name=None,  # Set per-query
                num_hops=num_hops
            )
            if multihop_model_path:
                print(f"Loading optimized model from: {multihop_model_path}")
                self.multihop_retriever.load(multihop_model_path)
        else:
            print("Using standard retrieval mode")
            self.query_reformulator = QueryReformulator(self.config.text_model)
            self.query_diversifier = QueryDiversifier(self.config.query_diversification, self.config.text_model)

        # Initialize generation components
        self.answerability_detector = AnswerabilityDetector(self.config.text_model)
        self.response_generator = ResponseGenerator(self.config.text_model)

        print("Pipeline initialized successfully!")
    
    def process_query(
        self,
        current_question: str,
        conversation_history: List[Dict[str, str]],
        collection_name: str,
        retrieval_only: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single query through the complete pipeline

        Args:
            current_question: User's current question (q)
            conversation_history: Full conversation history (C)
            collection_name: Document collection to search
            retrieval_only: If True, only run retrieval (phases 1-4), skip generation (phases 5-6)

        Returns:
            Complete response with metadata (R)
        """
        mode = "retrieval" if retrieval_only else "full pipeline"
        retrieval_mode = "multi-hop" if self.use_multihop else "standard"
        print(f"Processing query ({mode}, {retrieval_mode}): {current_question}")

        # ============================================================
        # RETRIEVAL: Multi-hop or Standard
        # ============================================================
        if self.use_multihop:
            # DSPy Multi-Hop Retrieval (replaces phases 1-4)
            print("Running DSPy Multi-Hop Retrieval...")
            multihop_result = self.multihop_retriever.process_query(
                current_question=current_question,
                conversation_history=conversation_history,
                collection_name=collection_name,
                retrieval_only=True
            )

            metadata = multihop_result["pipeline_metadata"]
            top_documents = metadata["retrieved_documents"]
            reformulated_query = metadata.get("reformulated_query", current_question)
            query_variants = metadata.get("query_variants", [])
            num_candidates = metadata.get("num_candidates", 0)

            print(f"Multi-hop retrieval complete: {len(top_documents)} documents from {metadata.get('num_hops', 3)} hops")

        else:
            # Standard Retrieval (phases 1-4)
            # Phase 1: Context-Aware Query Reformulation
            print("Phase 1: Query Reformulation...")
            reformulated_query = self.query_reformulator.cot_rewrite(
                current_question, conversation_history
            )
            print(f"Reformulated query: {reformulated_query}")

            # Phase 2: Multi-Strategy Query Diversification
            print("Phase 2: Query Diversification...")
            query_variants = self.query_diversifier.diversify(reformulated_query)
            print(f"Generated {len(query_variants)} query variants")

            # Phase 3: Hybrid Document Retrieval
            print("Phase 3: Hybrid Retrieval...")
            candidates = self.hybrid_retriever.hybrid_retrieve(collection_name, query_variants)
            num_candidates = len(candidates)

            # Phase 4: Cross-Encoder Reranking
            print("Phase 4: Reranking...")
            top_documents = self.hybrid_retriever.rerank_candidates(reformulated_query, candidates)
            print(f"Selected {len(top_documents)} top documents")

        # ============================================================
        # RETRIEVAL-ONLY MODE: Skip generation phases (5-6)
        # ============================================================
        if retrieval_only:
            print("Retrieval-only mode: Skipping answerability and generation phases")
            result = {
                "pipeline_metadata": {
                    "original_question": current_question,
                    "reformulated_query": reformulated_query,
                    "query_variants": query_variants,
                    "num_candidates": num_candidates,
                    "num_top_docs": len(top_documents),
                    "retrieved_documents": top_documents,
                    "retrieval_mode": retrieval_mode
                }
            }
            if self.use_multihop:
                result["pipeline_metadata"]["hop_details"] = metadata.get("hop_details", [])
            return result

        # ============================================================
        # FULL PIPELINE MODE: Run answerability and generation (5-6)
        # ============================================================

        # Phase 5: Answerability Classification
        print("Phase 5: Answerability Detection...")
        answerability = self.answerability_detector.classify(reformulated_query, top_documents)
        print(f"Answerability: {answerability.value}")

        # Phase 6: Conditional Response Generation
        print("Phase 6: Response Generation...")
        response = self.response_generator.generate_response(
            answerability,
            reformulated_query,
            top_documents[:5],  # Pass only top 5 docs for generation
            conversation_history
        )

        # Add pipeline metadata
        response.update({
            "pipeline_metadata": {
                "original_question": current_question,
                "reformulated_query": reformulated_query,
                "query_variants": query_variants,
                "answerability": answerability.value,
                "num_candidates": num_candidates,
                "num_top_docs": len(top_documents),
                "retrieved_documents": top_documents,
                "retrieval_mode": retrieval_mode
            }
        })

        if self.use_multihop:
            response["pipeline_metadata"]["hop_details"] = metadata.get("hop_details", [])

        print("Pipeline processing complete!")
        return response
    
    def setup_collection(self, collection_name: str, corpus_file: str, bm25_only: bool = False) -> bool:
        """
        Set up a document collection by indexing both vector and BM25

        Args:
            collection_name: Name for the collection
            corpus_file: Path to corpus JSONL file
            bm25_only: If True, only index BM25 (skip vector indexing)

        Returns:
            True if setup was successful
        """
        print(f"Setting up collection {collection_name}...")

        vector_success = True
        if not bm25_only:
            # Index for vector search
            print("Indexing for dense retrieval...")
            vector_success = self.vector_store.index_corpus(collection_name, corpus_file)
        else:
            print("Skipping dense retrieval indexing (BM25-only mode)")

        # Index for BM25 search
        print("Indexing for BM25 retrieval...")
        bm25_success = self.hybrid_retriever.index_corpus_bm25(collection_name, corpus_file)

        if vector_success and bm25_success:
            print(f"Collection {collection_name} setup complete!")
            return True
        else:
            print(f"Failed to setup collection {collection_name}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
        """
        return self.vector_store.get_collection_info(collection_name)
    
    def batch_process_queries(
        self,
        queries: List[Dict[str, Any]],
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of query dictionaries with 'text' and 'conversation_history'
            collection_name: Collection to search
            
        Returns:
            List of responses
        """
        results = []
        total_queries = len(queries)
        
        for i, query_item in enumerate(queries):
            if i % 10 == 0:
                print(f"Processing query {i+1}/{total_queries}...")
            
            try:
                response = self.process_query(
                    current_question=query_item['text'],
                    conversation_history=query_item.get('conversation_history', []),
                    collection_name=collection_name
                )
                
                # Add query metadata
                response['query_id'] = query_item.get('_id', f"query_{i}")
                results.append(response)
                
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                # Add error response
                error_response = {
                    'query_id': query_item.get('_id', f"query_{i}"),
                    'response_text': 'Sorry, I encountered an error processing your question.',
                    'confidence_score': 0.0,
                    'sources': [],
                    'guardrail_flags': {'error': str(e)},
                    'pipeline_metadata': {
                        'original_question': query_item['text'],
                        'error': str(e)
                    }
                }
                results.append(error_response)
        
        print(f"Batch processing complete! Processed {len(results)} queries")
        return results


def create_pipeline(
    config_path: str = None,
    use_multihop: bool = False,
    multihop_model_path: Optional[str] = None,
    num_hops: int = 3
) -> Pipeline:
    """
    Factory function to create a pipeline instance

    Args:
        config_path: Optional path to YAML configuration file
        use_multihop: If True, use DSPy multi-hop retrieval
        multihop_model_path: Path to optimized multi-hop model
        num_hops: Number of retrieval hops (default: 3)

    Returns:
        Initialized Pipeline instance
    """
    config = load_config(config_path)
    return Pipeline(
        config=config,
        use_multihop=use_multihop,
        multihop_model_path=multihop_model_path,
        num_hops=num_hops
    )
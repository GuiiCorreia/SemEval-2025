"""
DSPy Multi-Hop RAG Module

This module implements multi-hop retrieval using DSPy for query generation,
integrated with the existing hybrid retrieval system.

Architecture:
- QueryGenerator: Generates 3 search queries per hop
- NotesBuilder: Builds structured notes from retrieved context
- MultiHopRetriever: Orchestrates 3 hops of retrieval
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class GenerateQueries(dspy.Signature):
    """Generate 3 diverse search queries to retrieve relevant documents for answering the question."""

    question = dspy.InputField(desc="The user's question to answer")
    conversation_history = dspy.InputField(desc="Previous conversation turns for context")
    notes = dspy.InputField(desc="Accumulated notes from previous retrieval hops")

    query_1 = dspy.OutputField(desc="First search query")
    query_2 = dspy.OutputField(desc="Second search query")
    query_3 = dspy.OutputField(desc="Third search query")


class BuildNotes(dspy.Signature):
    """Analyze retrieved documents and build structured notes to guide the next retrieval hop."""

    question = dspy.InputField(desc="The user's question to answer")
    queries_used = dspy.InputField(desc="The 3 queries used in this hop")
    retrieved_documents = dspy.InputField(desc="The 10 reranked documents from this hop")
    previous_notes = dspy.InputField(desc="Notes from previous hops")

    key_findings = dspy.OutputField(desc="Key information found that helps answer the question")
    missing_info = dspy.OutputField(desc="What information is still needed to fully answer")
    search_suggestions = dspy.OutputField(desc="Suggestions for what to search next")


@dataclass
class HopResult:
    """Results from a single retrieval hop."""
    hop_number: int
    queries: List[str]
    candidates_count: int
    reranked_documents: List[Dict[str, Any]]
    notes: Optional[Dict[str, str]] = None


@dataclass
class MultiHopResult:
    """Complete results from multi-hop retrieval."""
    question: str
    conversation_history: List[Dict[str, str]]
    hop_results: List[HopResult] = field(default_factory=list)
    final_documents: List[Dict[str, Any]] = field(default_factory=list)
    all_queries: List[str] = field(default_factory=list)


class QueryGenerator(dspy.Module):
    """
    Generates 3 diverse search queries based on the question and accumulated context.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateQueries)

    def forward(
        self,
        question: str,
        conversation_history: str,
        notes: str
    ) -> dspy.Prediction:
        """
        Generate 3 search queries.

        Args:
            question: The user's question
            conversation_history: Formatted conversation history
            notes: Accumulated notes from previous hops

        Returns:
            Prediction with query_1, query_2, query_3
        """
        result = self.generate(
            question=question,
            conversation_history=conversation_history,
            notes=notes
        )

        return dspy.Prediction(
            queries=[result.query_1, result.query_2, result.query_3]
        )


class NotesBuilder(dspy.Module):
    """
    Builds structured notes from retrieved documents to guide next hop.
    """

    def __init__(self):
        super().__init__()
        self.build = dspy.ChainOfThought(BuildNotes)

    def forward(
        self,
        question: str,
        queries_used: List[str],
        retrieved_documents: List[Dict[str, Any]],
        previous_notes: str
    ) -> dspy.Prediction:
        """
        Build notes from retrieved context.

        Args:
            question: The user's question
            queries_used: List of 3 queries used in this hop
            retrieved_documents: The 10 reranked documents
            previous_notes: Notes from previous hops

        Returns:
            Prediction with key_findings, missing_info, search_suggestions
        """
        # Format queries
        queries_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries_used)])

        # Format documents
        docs_str = self._format_documents(retrieved_documents)

        result = self.build(
            question=question,
            queries_used=queries_str,
            retrieved_documents=docs_str,
            previous_notes=previous_notes
        )

        return dspy.Prediction(
            key_findings=result.key_findings,
            missing_info=result.missing_info,
            search_suggestions=result.search_suggestions
        )

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for the prompt."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Untitled')
            text = doc.get('text', '')[:800]  # Truncate long texts
            score = doc.get('final_score', doc.get('rerank_score', 0))
            formatted.append(f"[Doc {i}] (score: {score:.3f}) {title}\n{text}")

        return "\n\n".join(formatted)


class MultiHopRetriever(dspy.Module):
    """
    Orchestrates multi-hop retrieval with 3 hops.

    Each hop:
    1. QueryGenerator -> 3 queries
    2. HybridRetrieval(3 queries) -> 30 candidates
    3. Rerank(concat queries, candidates) -> 10 documents
    4. NotesBuilder -> notes for next hop
    """

    def __init__(
        self,
        hybrid_retriever,
        collection_name: str,
        num_hops: int = 3,
    ):
        """
        Initialize the multi-hop retriever.

        Args:
            hybrid_retriever: HybridRetriever instance from retrieval module
            collection_name: Name of the document collection
            num_hops: Number of retrieval hops (default: 3)
            candidates_per_query: Docs to retrieve per query before rerank (default: 10)
        """
        super().__init__()
        self.hybrid_retriever = hybrid_retriever
        self.collection_name = collection_name
        self.num_hops = num_hops

        # DSPy modules
        self.query_generator = QueryGenerator()
        self.notes_builder = NotesBuilder()

    def _format_conversation(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history as a string."""
        if not conversation_history:
            return "No previous conversation."

        formatted = []
        for msg in conversation_history:
            speaker = msg.get('speaker', 'unknown')
            text = msg.get('text', '')
            formatted.append(f"{speaker}: {text}")

        return "\n".join(formatted)

    def _format_notes(self, hop_results: List[HopResult]) -> str:
        """Format accumulated notes from all previous hops."""
        if not hop_results:
            return "No previous retrieval attempts. This is the first hop."

        notes_parts = []
        for hr in hop_results:
            hop_section = f"=== Hop {hr.hop_number} ===\n"
            hop_section += f"Queries used:\n"
            for i, q in enumerate(hr.queries, 1):
                hop_section += f"  {i}. {q}\n"

            if hr.notes:
                hop_section += f"\nKey findings: {hr.notes.get('key_findings', 'N/A')}\n"
                hop_section += f"Missing info: {hr.notes.get('missing_info', 'N/A')}\n"
                hop_section += f"Search suggestions: {hr.notes.get('search_suggestions', 'N/A')}\n"

            notes_parts.append(hop_section)

        return "\n".join(notes_parts)

    def _retrieve_and_rerank(
        self,
        queries: List[str]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Execute hybrid retrieval with 3 queries and rerank results.

        Args:
            queries: List of 3 search queries

        Returns:
            Tuple of (reranked_documents, total_candidates_count)
        """
        # Hybrid retrieval with all 3 queries -> ~30 candidates
        candidates = self.hybrid_retriever.hybrid_retrieve(
            self.collection_name,
            queries
        )

        candidates_count = len(candidates)

        # Rerank with concatenated queries as context
        rerank_query = " | ".join(queries)
        reranked = self.hybrid_retriever.rerank_candidates(rerank_query, candidates)

        # Return top documents
        return reranked, candidates_count

    def forward(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> dspy.Prediction:
        """
        Execute multi-hop retrieval.

        Args:
            question: The user's question
            conversation_history: Previous conversation turns

        Returns:
            Prediction with all results
        """
        conversation_history = conversation_history or []
        formatted_history = self._format_conversation(conversation_history)

        result = MultiHopResult(
            question=question,
            conversation_history=conversation_history
        )

        print("User question: ", question)
        print("Conversation History: ", formatted_history)

        # Execute hops
        for hop_num in range(1, self.num_hops + 1):
            print(f"\n{'='*50}")
            print(f"HOP {hop_num}/{self.num_hops}")
            print(f"{'='*50}")

            try:
                # Get accumulated notes
                notes_str = self._format_notes(result.hop_results)

                # Generate 3 queries
                print("Generating queries...")
                query_result = self.query_generator(
                    question=question,
                    conversation_history=formatted_history,
                    notes=notes_str
                )
                queries = query_result.queries

                # Filter out None queries
                valid_queries = [q for q in queries if q is not None]
                if not valid_queries:
                    print(f"  Warning: No valid queries generated, skipping hop {hop_num}")
                    continue

                queries = valid_queries
                result.all_queries.extend(queries)

                for i, q in enumerate(queries, 1):
                    print(f"  Query {i}: {q}")

                # Retrieve and rerank
                print("Retrieving and reranking...")
                reranked_docs, candidates_count = self._retrieve_and_rerank(queries)
                print(f"  Candidates: {candidates_count} -> Reranked: {len(reranked_docs)}")

                # Build notes (except for last hop)
                notes_dict = None
                if hop_num < self.num_hops:
                    print("Building notes...")
                    notes_result = self.notes_builder(
                        question=question,
                        queries_used=queries,
                        retrieved_documents=reranked_docs,
                        previous_notes=notes_str
                    )
                    notes_dict = {
                        'key_findings': notes_result.key_findings,
                        'missing_info': notes_result.missing_info,
                        'search_suggestions': notes_result.search_suggestions
                    }
                    print(f"  Key findings: {notes_result.key_findings}")

                # Store hop result
                hop_result = HopResult(
                    hop_number=hop_num,
                    queries=queries,
                    candidates_count=candidates_count,
                    reranked_documents=reranked_docs,
                    notes=notes_dict
                )
                result.hop_results.append(hop_result)

            except Exception as e:
                print(f"  Error in hop {hop_num}: {e}, continuing with existing documents...")

        # Aggregate final documents (deduplicated from all hops)
        aggregated_docs = self._aggregate_documents(result.hop_results)
        print(f"\nTotal unique documents: {len(aggregated_docs)}")

        # Final reranking using all generated queries
        if aggregated_docs and result.all_queries:
            print("Final reranking...")
            rerank_query = " | ".join(result.all_queries)
            reranked_final = self.hybrid_retriever.rerank_candidates(rerank_query, aggregated_docs)
            result.final_documents = reranked_final
            print(f"Final documents after rerank: {len(result.final_documents)}")
        elif aggregated_docs:
            # No queries but have docs - use original question for rerank
            print("Final reranking with original question...")
            reranked_final = self.hybrid_retriever.rerank_candidates(question, aggregated_docs)
            result.final_documents = reranked_final
            print(f"Final documents after rerank: {len(result.final_documents)}")
        else:
            print("No documents retrieved from any hop")
            result.final_documents = []

        return dspy.Prediction(
            question=result.question,
            all_queries=result.all_queries,
            hop_results=result.hop_results,
            final_documents=result.final_documents,
            num_hops=len(result.hop_results)
        )

    def _aggregate_documents(
        self,
        hop_results: List[HopResult]
    ) -> List[Dict[str, Any]]:
        """Aggregate and deduplicate documents from all hops."""
        seen_ids = set()
        all_docs = []

        # Process in reverse order (later hops likely have more refined results)
        for hr in reversed(hop_results):
            for doc in hr.reranked_documents:
                doc_id = doc['document_id']
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        return all_docs

    def process_query(
        self,
        current_question: str,
        conversation_history: List[Dict[str, str]] = None,
        collection_name: str = None,
        retrieval_only: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query and return results in the same format as Pipeline.process_query.

        This method provides compatibility with the original pipeline interface.

        Args:
            current_question: The user's current question
            conversation_history: Previous conversation turns
            collection_name: Document collection (overrides instance collection if provided)
            retrieval_only: If True, only return retrieval results (default: True)

        Returns:
            Dictionary with pipeline_metadata in the same format as Pipeline.process_query
        """
        # Use provided collection or fall back to instance collection
        effective_collection = collection_name or self.collection_name

        # Temporarily override collection for this query
        original_collection = self.collection_name
        self.collection_name = effective_collection

        # Run multi-hop retrieval
        prediction = self.forward(
            question=current_question,
            conversation_history=conversation_history or []
        )

        # Restore original collection
        self.collection_name = original_collection

        # Calculate total candidates across all hops
        total_candidates = sum(hr.candidates_count for hr in prediction.hop_results)

        # Format as pipeline-compatible output
        return {
            "pipeline_metadata": {
                "original_question": current_question,
                "reformulated_query": prediction.all_queries[0] if prediction.all_queries else current_question,
                "query_variants": prediction.all_queries,
                "num_candidates": total_candidates,
                "num_top_docs": len(prediction.final_documents),
                "retrieved_documents": prediction.final_documents,
                # Multi-hop specific metadata
                "num_hops": prediction.num_hops,
                "hop_details": [
                    {
                        "hop": hr.hop_number,
                        "queries": hr.queries,
                        "candidates": hr.candidates_count,
                        "docs_after_rerank": len(hr.reranked_documents),
                        "notes": hr.notes
                    }
                    for hr in prediction.hop_results
                ]
            }
        }


def create_multihop_retriever(
    hybrid_retriever,
    collection_name: str = None,
    num_hops: int = 3
) -> MultiHopRetriever:
    """
    Factory function to create a MultiHopRetriever.

    Args:
        hybrid_retriever: HybridRetriever instance
        collection_name: Default collection name (can be overridden per-query via process_query)
        num_hops: Number of hops (default: 3)

    Returns:
        Configured MultiHopRetriever
    """
    return MultiHopRetriever(
        hybrid_retriever=hybrid_retriever,
        collection_name=collection_name,
        num_hops=num_hops,
    )
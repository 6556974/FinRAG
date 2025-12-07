"""
Retrieval system for FinRAG
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from .data_loader import Document, Query
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval"""
    query_id: str
    query_text: str
    retrieved_docs: List[Tuple[str, float]]  # (doc_id, score)
    
    def get_top_k(self, k: int) -> List[str]:
        """Get top k document IDs"""
        return [doc_id for doc_id, _ in self.retrieved_docs[:k]]
    
    def get_top_k_with_scores(self, k: int) -> List[Tuple[str, float]]:
        """Get top k documents with scores"""
        return self.retrieved_docs[:k]


class Retriever:
    """Document retrieval system"""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        documents: Dict[str, Document],
        top_k: int = 10
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            documents: Dictionary of documents
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k
        
        logger.info(f"Initialized retriever with {len(documents)} documents")
    
    def retrieve(
        self, 
        query_id: str,
        query_text: str,
        query_embedding,
        top_k: int = None
    ) -> RetrievalResult:
        """
        Retrieve documents for a query
        
        Args:
            query_id: Query ID
            query_text: Query text
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve (uses self.top_k if None)
            
        Returns:
            RetrievalResult object
        """
        k = top_k if top_k is not None else self.top_k
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k)
        
        return RetrievalResult(
            query_id=query_id,
            query_text=query_text,
            retrieved_docs=results
        )
    
    def batch_retrieve(
        self,
        queries: Dict[str, Query],
        query_embeddings: Dict[str, any],
        top_k: int = None
    ) -> Dict[str, RetrievalResult]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: Dictionary of Query objects
            query_embeddings: Dictionary of query embeddings
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dictionary mapping query IDs to RetrievalResult objects
        """
        results = {}
        
        for query_id, query in queries.items():
            if query_id not in query_embeddings:
                logger.warning(f"No embedding found for query {query_id}")
                continue
            
            results[query_id] = self.retrieve(
                query_id=query_id,
                query_text=query.text,
                query_embedding=query_embeddings[query_id],
                top_k=top_k
            )
        
        logger.info(f"Retrieved documents for {len(results)} queries")
        return results
    
    def get_retrieved_documents(
        self, 
        retrieval_result: RetrievalResult,
        include_scores: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Get full Document objects for retrieval results
        
        Args:
            retrieval_result: RetrievalResult object
            include_scores: Whether to include scores
            
        Returns:
            List of Document objects or (Document, score) tuples
        """
        if include_scores:
            return [
                (self.documents[doc_id], score)
                for doc_id, score in retrieval_result.retrieved_docs
                if doc_id in self.documents
            ]
        else:
            return [
                self.documents[doc_id]
                for doc_id, _ in retrieval_result.retrieved_docs
                if doc_id in self.documents
            ]
    
    def format_context(
        self, 
        retrieval_result: RetrievalResult,
        max_contexts: int = 5,
        include_titles: bool = True
    ) -> str:
        """
        Format retrieved documents as context string for LLM
        
        Args:
            retrieval_result: RetrievalResult object
            max_contexts: Maximum number of contexts to include
            include_titles: Whether to include document titles
            
        Returns:
            Formatted context string
        """
        docs_with_scores = self.get_retrieved_documents(
            retrieval_result, 
            include_scores=True
        )[:max_contexts]
        
        context_parts = []
        
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            if include_titles and doc.title:
                context_parts.append(f"Document {i} (Relevance: {score:.3f}):")
                context_parts.append(f"Title: {doc.title}")
                context_parts.append(f"Content: {doc.text}")
            else:
                context_parts.append(f"Document {i} (Relevance: {score:.3f}):")
                context_parts.append(doc.text)
            
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)


def evaluate_retrieval(
    retrieval_results: Dict[str, RetrievalResult],
    qrels: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    """
    Evaluate retrieval performance
    
    Args:
        retrieval_results: Dictionary of retrieval results
        qrels: Ground truth relevance judgments
        k_values: K values for Recall@K and Precision@K
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics["mrr"] = []  # Mean Reciprocal Rank
    
    for query_id, result in retrieval_results.items():
        if query_id not in qrels:
            continue
        
        relevant_docs = set(qrels[query_id])
        retrieved_ids = result.get_top_k(max(k_values))
        
        # Calculate metrics for each k
        for k in k_values:
            top_k = retrieved_ids[:k]
            relevant_in_k = len(set(top_k) & relevant_docs)
            
            # Recall@K
            recall = relevant_in_k / len(relevant_docs) if relevant_docs else 0
            metrics[f"recall@{k}"].append(recall)
            
            # Precision@K
            precision = relevant_in_k / k if k > 0 else 0
            metrics[f"precision@{k}"].append(precision)
        
        # MRR
        reciprocal_rank = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                reciprocal_rank = 1.0 / rank
                break
        metrics["mrr"].append(reciprocal_rank)
    
    # Average metrics
    averaged_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            averaged_metrics[metric_name] = sum(values) / len(values)
        else:
            averaged_metrics[metric_name] = 0.0
    
    return averaged_metrics


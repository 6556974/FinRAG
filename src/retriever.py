"""
Retrieval system for FinRAG
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
import json

import yaml
import boto3
from rank_bm25 import BM25Okapi

from .data_loader import Document, Query
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval"""
    query_id: str
    query_text: str
    # List of (doc_id, score)
    retrieved_docs: List[Tuple[str, float]]

    def get_top_k(self, k: int) -> List[str]:
        """Get top-k document IDs"""
        return [doc_id for doc_id, _ in self.retrieved_docs[:k]]

    def get_top_k_with_scores(self, k: int) -> List[Tuple[str, float]]:
        """Get top-k documents with scores"""
        return self.retrieved_docs[:k]


class Retriever:
    """Document retrieval system"""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        documents: Dict[str, Document],
        top_k: int = 10,
    ):
        """
        Initialize retriever

        Args:
            vector_store: Vector store containing document embeddings
            documents: Dictionary of documents (doc_id -> Document)
            top_k: Default number of documents to retrieve
        """
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k

        # Keep stable ordering of doc_ids for BM25
        self.doc_ids: List[str] = list(self.documents.keys())

        # --- Load retrieval config from config.yaml (single place of truth) ---
        try:
            with open("config.yaml", "r") as f:
                _cfg = yaml.safe_load(f)
            retrieval_cfg = _cfg.get("retrieval", {})
        except Exception as e:
            logger.warning(
                f"Failed to load retrieval config from config.yaml, "
                f"falling back to defaults. Error: {e}"
            )
            retrieval_cfg = {}

        self.use_hybrid: bool = retrieval_cfg.get("hybrid_search", False)
        self.use_rerank: bool = retrieval_cfg.get("rerank", False)

        # --- BM25 setup (for hybrid search) ---
        if self.use_hybrid:
            tokenized_docs = [self.documents[doc_id].text.split()
                              for doc_id in self.doc_ids]
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(
                f"BM25 enabled for hybrid search over {len(self.doc_ids)} documents"
            )
        else:
            self.bm25 = None

        # --- Titan Rerank client (only if enabled) ---
        if self.use_rerank:
            self.rerank_client = boto3.client(
                "bedrock-runtime", region_name="us-west-2"
            )
            logger.info("Titan Rerank enabled (amazon.rerank-v1:0)")
        else:
            self.rerank_client = None

        logger.info(
            f"Initialized retriever with {len(documents)} documents "
            f"(hybrid={self.use_hybrid}, rerank={self.use_rerank})"
        )

    # ---------------------------------------------------------------------
    # Core retrieval logic
    # ---------------------------------------------------------------------
    def retrieve(
        self,
        query_id: str,
        query_text: str,
        query_embedding,
        top_k: int = None,
    ) -> RetrievalResult:
        """
        Retrieve documents for a query (vector / hybrid / rerank).

        Returns RetrievalResult where:
        - retrieved_docs is List[(doc_id, score)]
        - score = vector similarity or rerank score (depending on pipeline)
        """
        k = top_k if top_k is not None else self.top_k

        # 1) Vector search (baseline using FAISS)
        embedding_results: List[Tuple[str, float]] = self.vector_store.search(
            query_embedding, k
        )
        # embedding_results: [(doc_id, score), ...]

        # If no hybrid and no rerank, just return baseline results
        if not self.use_hybrid and not self.use_rerank:
            return RetrievalResult(
                query_id=query_id,
                query_text=query_text,
                retrieved_docs=embedding_results,
            )

        # -----------------------------------------------------------------
        # 2) Hybrid: add BM25 docs (union of doc_ids, scores kept simple)
        # -----------------------------------------------------------------
        combined_scores: Dict[str, float] = {}

        # Start with vector scores (keep FAISS ranking as primary)
        for doc_id, score in embedding_results:
            combined_scores[doc_id] = float(score)

        if self.use_hybrid and self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(query_text.split())
            # Rank all docs by BM25
            ranked_indices = sorted(
                range(len(self.doc_ids)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )
            # Take top-k BM25 docs
            for idx in ranked_indices[:k]:
                doc_id = self.doc_ids[idx]
                score = float(bm25_scores[idx])
                # If vector already has this doc, keep vector score
                combined_scores.setdefault(doc_id, score)

        # At this point combined_scores contains candidate doc_ids
        # with some score (vector or BM25). Order is insertion order:
        #  - all vector docs first in FAISS order
        #  - then BM25-only docs
        candidate_doc_ids_scores: List[Tuple[str, float]] = list(
            combined_scores.items()
        )

        # -----------------------------------------------------------------
        # 3) Titan Rerank (optional): reorder candidates using LLM reranker
        # -----------------------------------------------------------------
        if self.use_rerank and self.rerank_client is not None:
            try:
                # Prepare candidate texts in the same order
                candidate_texts = [
                    self.documents[doc_id].text for doc_id, _ in candidate_doc_ids_scores
                ]

                payload = {
                    "query": query_text,
                    "documents": [{"text": t} for t in candidate_texts]
                }

                resp = self.rerank_client.invoke_model(
                    modelId="amazon.rerank-v1:0",
                    body=json.dumps(payload),
                    contentType="application/json",
                    accept="application/json",
                )
                data = json.loads(resp["body"].read())
                results = data.get("results", [])
                # results: [{"index": int, "score": float}, ...]

                # Rebuild (doc_id, score) according to rerank score
                reranked: List[Tuple[str, float]] = []
                for item in sorted(
                    results, key=lambda x: x["relevance_score"], reverse=True
                ):
                    idx = item["index"]
                    if 0 <= idx < len(candidate_doc_ids_scores):
                        doc_id = candidate_doc_ids_scores[idx][0]
                        score = float(item["relevance_score"])
                        reranked.append((doc_id, score))

                # Fallback: if rerank returns nothing, keep original
                if reranked:
                    candidate_doc_ids_scores = reranked

            except Exception as e:
                logger.error(f"Rerank failed for query {query_id}: {e}")
                # Fallback to vector/BM25 order

        # Finally, take top-k
        final_docs = candidate_doc_ids_scores[:k]

        return RetrievalResult(
            query_id=query_id,
            query_text=query_text,
            retrieved_docs=final_docs,
        )

    # ---------------------------------------------------------------------
    # Batch retrieval
    # ---------------------------------------------------------------------
    def batch_retrieve(
        self,
        queries: Dict[str, Query],
        query_embeddings: Dict[str, any],
        top_k: int = None,
    ) -> Dict[str, RetrievalResult]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: Dict of Query objects (query_id -> Query)
            query_embeddings: Dict of embeddings (query_id -> embedding vector)
            top_k: Number of documents to retrieve per query

        Returns:
            Dict mapping query_id -> RetrievalResult
        """
        results: Dict[str, RetrievalResult] = {}

        for query_id, query in queries.items():
            if query_id not in query_embeddings:
                logger.warning(f"No embedding found for query {query_id}")
                continue

            results[query_id] = self.retrieve(
                query_id=query_id,
                query_text=query.text,
                query_embedding=query_embeddings[query_id],
                top_k=top_k,
            )

        logger.info(f"Retrieved documents for {len(results)} queries")
        return results

    # ---------------------------------------------------------------------
    # Helpers used by RAG pipeline
    # ---------------------------------------------------------------------
    def get_retrieved_documents(
        self,
        retrieval_result: RetrievalResult,
        include_scores: bool = False,
    ):
        """
        Get full Document objects for retrieval results.

        If include_scores=True, returns List[(Document, score)].
        Else, returns List[Document].
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
        include_titles: bool = True,
    ) -> str:
        """
        Format retrieved documents as context string for LLM.

        Args:
            retrieval_result: RetrievalResult object
            max_contexts: Maximum number of contexts to include
            include_titles: Whether to include document titles

        Returns:
            Formatted context string
        """
        docs_with_scores = self.get_retrieved_documents(
            retrieval_result, include_scores=True
        )[:max_contexts]

        context_parts: List[str] = []

        for i, (doc, score) in enumerate(docs_with_scores, 1):
            context_parts.append(f"Document {i} (Relevance: {score:.3f}):")
            if include_titles and doc.title:
                context_parts.append(f"Title: {doc.title}")
                context_parts.append(f"Content: {doc.text}")
            else:
                context_parts.append(doc.text)
            context_parts.append("")  # Blank line between docs

        return "\n".join(context_parts)


# -------------------------------------------------------------------------
# Standalone retrieval evaluation (kept for backward compatibility)
# -------------------------------------------------------------------------
def evaluate_retrieval(
    retrieval_results: Dict[str, RetrievalResult],
    qrels: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using Recall@K, Precision@K, and MRR.

    Args:
        retrieval_results: Dict mapping query_id -> RetrievalResult
        qrels: Dict mapping query_id -> list of relevant doc_ids
        k_values: List of K values to compute metrics for

    Returns:
        Dict containing averaged metrics:
        {
            "recall@1": float,
            "precision@1": float,
            "mrr": float,
            ...
        }
    """

    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for query_id, result in retrieval_results.items():
        if query_id not in qrels:
            continue

        relevant = set(qrels[query_id])
        retrieved_ids = result.get_top_k(max(k_values))

        # Recall@K & Precision@K
        for k in k_values:
            top_k_docs = retrieved_ids[:k]
            correct = len(set(top_k_docs) & relevant)

            recall = correct / len(relevant) if len(relevant) > 0 else 0.0
            precision = correct / k if k > 0 else 0.0

            metrics[f"recall@{k}"].append(recall)
            metrics[f"precision@{k}"].append(precision)

        # MRR
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        metrics["mrr"].append(rr)

    # Average
    averaged: Dict[str, float] = {}
    for name, values in metrics.items():
        averaged[name] = sum(values) / len(values) if values else 0.0

    return averaged

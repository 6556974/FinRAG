"""
Vector store implementation using FAISS
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(
        self,
        dimension: int = 1024,
        metric: str = "cosine",
        index_path: str = None
    ):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric ("cosine", "euclidean", "dot_product")
            index_path: Path to save/load index
        """
        self.dimension = dimension
        self.metric = metric
        self.index_path = Path(index_path) if index_path else None
        
        # Initialize FAISS index
        if metric == "cosine":
            # Use inner product with normalized vectors for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = True
        elif metric == "euclidean":
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        elif metric == "dot_product":
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = False
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Mapping from index position to document ID
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        logger.info(f"Initialized FAISS index with {metric} metric")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        if not self.normalize:
            return vectors
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add_documents(self, doc_embeddings: Dict[str, np.ndarray]):
        """
        Add documents to the vector store
        
        Args:
            doc_embeddings: Dictionary mapping document IDs to embeddings
        """
        doc_ids = list(doc_embeddings.keys())
        embeddings = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])
        
        # Normalize if using cosine similarity
        embeddings = self._normalize_vectors(embeddings)
        
        # Get starting index
        start_idx = len(self.id_to_index)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Update mappings
        for i, doc_id in enumerate(doc_ids):
            idx = start_idx + i
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
        
        logger.info(f"Added {len(doc_ids)} documents to index. Total: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if using cosine similarity
        query_embedding = self._normalize_vectors(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Convert to document IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                doc_id = self.index_to_id.get(idx)
                if doc_id:
                    results.append((doc_id, float(score)))
        
        return results
    
    def batch_search(
        self, 
        query_embeddings: Dict[str, np.ndarray], 
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embeddings
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping query IDs to lists of (document_id, score) tuples
        """
        results = {}
        
        for query_id, embedding in query_embeddings.items():
            results[query_id] = self.search(embedding, top_k)
        
        return results
    
    def save(self, path: str = None):
        """
        Save index to disk
        
        Args:
            path: Path to save index (uses self.index_path if not provided)
        """
        save_path = Path(path) if path else self.index_path
        
        if not save_path:
            raise ValueError("No save path provided")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save mappings
        mappings_file = save_path / "mappings.pkl"
        with open(mappings_file, 'wb') as f:
            pickle.dump({
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'dimension': self.dimension,
                'metric': self.metric
            }, f)
        
        logger.info(f"Saved index to {save_path}")
    
    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """
        Load index from disk
        
        Args:
            path: Path to load index from
            
        Returns:
            FAISSVectorStore instance
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")
        
        # Load mappings
        mappings_file = load_path / "mappings.pkl"
        with open(mappings_file, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            dimension=data['dimension'],
            metric=data['metric'],
            index_path=str(load_path)
        )
        
        # Load FAISS index
        index_file = load_path / "index.faiss"
        instance.index = faiss.read_index(str(index_file))
        
        # Restore mappings
        instance.id_to_index = data['id_to_index']
        instance.index_to_id = data['index_to_id']
        
        logger.info(f"Loaded index from {load_path} with {instance.index.ntotal} documents")
        
        return instance
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_documents': self.index.ntotal,
            'dimension': self.dimension,
            'metric': self.metric,
            'index_type': type(self.index).__name__
        }


"""
Embedding generation using AWS Bedrock
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class BedrockEmbeddings:
    """AWS Bedrock embeddings wrapper"""
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimension: int = 1024,
        batch_size: int = 32,
        cache_dir: str = None
    ):
        """
        Initialize Bedrock embeddings
        
        Args:
            model_id: Bedrock embedding model ID
            region: AWS region
            dimension: Embedding dimension
            batch_size: Batch size for processing
            cache_dir: Directory to cache embeddings
        """
        self.model_id = model_id
        self.dimension = dimension
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize Bedrock client
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )
            logger.info(f"Initialized Bedrock client in region {region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Cache for embeddings
        self._cache: Dict[str, np.ndarray] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _load_cache(self):
        """Load embeddings cache from disk"""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save embeddings cache to disk"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text using Bedrock
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]
        
        try:
            # Prepare request body based on model
            if "amazon.titan-embed" in self.model_id:
                request_body = json.dumps({
                    "inputText": text[:8000]  # Titan has 8K token limit
                })
            elif "cohere.embed" in self.model_id:
                request_body = json.dumps({
                    "texts": [text],
                    "input_type": "search_document"
                })
            else:
                raise ValueError(f"Unsupported embedding model: {self.model_id}")
            
            # Call Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if "amazon.titan-embed" in self.model_id:
                embedding = np.array(response_body['embedding'], dtype=np.float32)
            elif "cohere.embed" in self.model_id:
                embedding = np.array(response_body['embeddings'][0], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported model response format")
            
            # Cache the result
            self._cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def embed_texts(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [len(texts), dimension])
        """
        embeddings = []
        
        iterator = tqdm(texts, desc="Generating embeddings") if show_progress else texts
        
        for text in iterator:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        # Save cache periodically
        if len(self._cache) % 100 == 0:
            self._save_cache()
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_documents(self, documents: Dict) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for documents
        
        Args:
            documents: Dictionary of Document objects
            
        Returns:
            Dictionary mapping document IDs to embeddings
        """
        doc_ids = list(documents.keys())
        
        # Prepare texts: combine title and text
        texts = []
        for doc_id in doc_ids:
            doc = documents[doc_id]
            # Create a combined text representation
            if doc.title:
                combined = f"{doc.title}\n\n{doc.text}"
            else:
                combined = doc.text
            texts.append(combined)
        
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embed_texts(texts, show_progress=True)
        
        # Create mapping
        doc_embeddings = {
            doc_id: embedding 
            for doc_id, embedding in zip(doc_ids, embeddings)
        }
        
        # Save cache
        self._save_cache()
        
        return doc_embeddings
    
    def embed_queries(self, queries: Dict) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for queries
        
        Args:
            queries: Dictionary of Query objects
            
        Returns:
            Dictionary mapping query IDs to embeddings
        """
        query_ids = list(queries.keys())
        texts = [queries[qid].text for qid in query_ids]
        
        logger.info(f"Generating embeddings for {len(texts)} queries")
        embeddings = self.embed_texts(texts, show_progress=True)
        
        # Create mapping
        query_embeddings = {
            qid: embedding 
            for qid, embedding in zip(query_ids, embeddings)
        }
        
        # Save cache
        self._save_cache()
        
        return query_embeddings
    
    def __del__(self):
        """Save cache on deletion"""
        self._save_cache()


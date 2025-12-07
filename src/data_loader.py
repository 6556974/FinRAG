"""
Data loading utilities for FinRAG system
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation"""
    id: str
    title: str
    text: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Query:
    """Query representation"""
    id: str
    text: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataLoader:
    """Loads and manages financial datasets"""
    
    def __init__(self, base_path: str):
        """
        Initialize DataLoader
        
        Args:
            base_path: Base path to the data directory
        """
        self.base_path = Path(base_path)
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data directory not found: {base_path}")
    
    def load_corpus(self, corpus_file: str) -> Dict[str, Document]:
        """
        Load corpus documents from JSONL file
        
        Args:
            corpus_file: Path to corpus JSONL file (relative to base_path)
            
        Returns:
            Dictionary mapping document IDs to Document objects
        """
        corpus_path = self.base_path / corpus_file
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        documents = {}
        
        logger.info(f"Loading corpus from {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        id=data['_id'],
                        title=data.get('title', ''),
                        text=data['text'],
                        metadata={'line_num': line_num}
                    )
                    documents[doc.id] = doc
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_queries(self, queries_file: str) -> Dict[str, Query]:
        """
        Load queries from JSONL file
        
        Args:
            queries_file: Path to queries JSONL file (relative to base_path)
            
        Returns:
            Dictionary mapping query IDs to Query objects
        """
        queries_path = self.base_path / queries_file
        
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        
        queries = {}
        
        logger.info(f"Loading queries from {queries_path}")
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    query = Query(
                        id=data['_id'],
                        text=data['text'],
                        metadata={'line_num': line_num, 'title': data.get('title', '')}
                    )
                    queries[query.id] = query
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def load_qrels(self, qrels_file: str) -> Dict[str, List[str]]:
        """
        Load query relevance judgments (ground truth)
        
        Args:
            qrels_file: Path to qrels TSV file (relative to base_path)
            
        Returns:
            Dictionary mapping query IDs to lists of relevant document IDs
        """
        qrels_path = self.base_path / qrels_file
        
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
        
        logger.info(f"Loading qrels from {qrels_path}")
        
        # Read TSV file
        df = pd.read_csv(qrels_path, sep='\t')
        
        # Group by query_id
        qrels = {}
        for query_id, group in df.groupby('query_id'):
            qrels[query_id] = group['corpus_id'].tolist()
        
        logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels
    
    def load_dataset(
        self, 
        corpus_file: str, 
        queries_file: str, 
        qrels_file: str
    ) -> Tuple[Dict[str, Document], Dict[str, Query], Dict[str, List[str]]]:
        """
        Load a complete dataset (corpus, queries, and qrels)
        
        Args:
            corpus_file: Path to corpus JSONL file
            queries_file: Path to queries JSONL file
            qrels_file: Path to qrels TSV file
            
        Returns:
            Tuple of (documents, queries, qrels)
        """
        documents = self.load_corpus(corpus_file)
        queries = self.load_queries(queries_file)
        qrels = self.load_qrels(qrels_file)
        
        return documents, queries, qrels
    
    def load_all_datasets(
        self, 
        dataset_configs: List[Dict]
    ) -> Dict[str, Tuple[Dict[str, Document], Dict[str, Query], Dict[str, List[str]]]]:
        """
        Load multiple datasets
        
        Args:
            dataset_configs: List of dataset configuration dictionaries
            
        Returns:
            Dictionary mapping dataset names to (documents, queries, qrels) tuples
        """
        datasets = {}
        
        for config in dataset_configs:
            name = config['name']
            logger.info(f"Loading dataset: {name}")
            
            try:
                datasets[name] = self.load_dataset(
                    corpus_file=config['corpus'],
                    queries_file=config['queries'],
                    qrels_file=config['qrels']
                )
            except Exception as e:
                logger.error(f"Error loading dataset {name}: {e}")
                continue
        
        return datasets


def merge_corpora(datasets: Dict) -> Dict[str, Document]:
    """
    Merge corpus documents from multiple datasets
    
    Args:
        datasets: Dictionary of datasets from load_all_datasets
        
    Returns:
        Merged dictionary of all documents
    """
    all_documents = {}
    
    for dataset_name, (documents, _, _) in datasets.items():
        logger.info(f"Merging {len(documents)} documents from {dataset_name}")
        all_documents.update(documents)
    
    logger.info(f"Total merged documents: {len(all_documents)}")
    return all_documents


def merge_queries(datasets: Dict) -> Dict[str, Query]:
    """
    Merge queries from multiple datasets
    
    Args:
        datasets: Dictionary of datasets from load_all_datasets
        
    Returns:
        Merged dictionary of all queries
    """
    all_queries = {}
    
    for dataset_name, (_, queries, _) in datasets.items():
        logger.info(f"Merging {len(queries)} queries from {dataset_name}")
        # Add dataset name to metadata
        for qid, query in queries.items():
            query.metadata['dataset'] = dataset_name
            all_queries[qid] = query
    
    logger.info(f"Total merged queries: {len(all_queries)}")
    return all_queries


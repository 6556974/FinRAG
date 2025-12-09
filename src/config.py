"""
Configuration management for FinRAG system
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AWSConfig(BaseModel):
    """AWS Bedrock configuration"""
    region: str = Field(default="us-east-1")
    embedding_model: str = Field(default="amazon.titan-embed-text-v2:0")
    llm_model: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)


class GeminiConfig(BaseModel):
    """Google Gemini configuration"""
    model_id: str = Field(default="gemini-3-pro-preview")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)


class DatasetConfig(BaseModel):
    """Dataset configuration"""
    name: str
    corpus: str
    queries: str
    qrels: str


class DataConfig(BaseModel):
    """Data paths configuration"""
    base_path: str
    datasets: List[DatasetConfig]
    output_dir: str = Field(default="outputs")
    cache_dir: str = Field(default="cache")


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    type: str = Field(default="faiss")
    index_path: str = Field(default="cache/faiss_index")
    dimension: int = Field(default=1024)
    metric: str = Field(default="cosine")


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = Field(default=10)
    rerank: bool = Field(default=False)
    hybrid_search: bool = Field(default=False)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)


class RAGConfig(BaseModel):
    """RAG pipeline configuration"""
    context_window: int = Field(default=8000)
    max_contexts: int = Field(default=5)
    prompt_template: str


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: List[str]
    sample_size: int | None = None
    ragas_llm: str = Field(default="gpt-4")
    ragas_embeddings: str = Field(default="openai")


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    batch_size: int = Field(default=32)
    max_workers: int = Field(default=4)
    cache_embeddings: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class"""
    llm_provider: str = Field(default="bedrock")
    aws: AWSConfig
    gemini: GeminiConfig
    data: DataConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    rag: RAGConfig
    evaluation: EvaluationConfig
    performance: PerformanceConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        aws_config = AWSConfig(**config_dict["aws"]["bedrock"])
        aws_config.region = config_dict["aws"]["region"]
        
        gemini_data = config_dict.get("gemini", {})
        gemini_config = GeminiConfig(**gemini_data) if gemini_data else GeminiConfig()
        
        dataset_configs = [
            DatasetConfig(**ds) for ds in config_dict["data"]["datasets"]
        ]
        
        data_config = DataConfig(
            base_path=config_dict["data"]["base_path"],
            datasets=dataset_configs,
            output_dir=config_dict["data"]["output_dir"],
            cache_dir=config_dict["data"]["cache_dir"]
        )
        
        return cls(
            llm_provider=config_dict.get("llm_provider", "bedrock"),
            aws=aws_config,
            gemini=gemini_config,
            data=data_config,
            vector_store=VectorStoreConfig(**config_dict["vector_store"]),
            retrieval=RetrievalConfig(**config_dict["retrieval"]),
            rag=RAGConfig(**config_dict["rag"]),
            evaluation=EvaluationConfig(**config_dict["evaluation"]),
            performance=PerformanceConfig(**config_dict["performance"])
        )


def load_config(config_path: str = "config.yaml") -> Config:
    """Convenience function to load configuration"""
    return Config.from_yaml(config_path)


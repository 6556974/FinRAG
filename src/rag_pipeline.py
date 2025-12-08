"""
RAG Pipeline using AWS Bedrock LLMs
"""

import boto3
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG system response"""
    query_id: str
    query_text: str
    answer: str
    contexts: List[str]
    context_ids: List[str]
    retrieval_scores: List[float]
    metadata: Dict = None


class BedrockLLM:
    """AWS Bedrock LLM wrapper"""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str = "us-east-1",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.9
    ):
        """
        Initialize Bedrock LLM
        
        Args:
            model_id: Bedrock model ID
            region: AWS region
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize Bedrock client
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )
            logger.info(f"Initialized Bedrock LLM: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using Bedrock LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            # Prepare request based on model family
            if "anthropic.claude" in self.model_id:
                request_body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            elif "amazon.titan" in self.model_id:
                request_body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.max_tokens,
                        "temperature": self.temperature,
                        "topP": self.top_p
                    }
                })
            elif "meta.llama" in self.model_id:
                request_body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                })
            else:
                raise ValueError(f"Unsupported model: {self.model_id}")
            
            # Call Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if "anthropic.claude" in self.model_id:
                return response_body['content'][0]['text']
            elif "amazon.titan" in self.model_id:
                return response_body['results'][0]['outputText']
            elif "meta.llama" in self.model_id:
                return response_body['generation']
            else:
                raise ValueError(f"Unsupported model response format")
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"


class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(
        self,
        retriever: Retriever,
        llm: BedrockLLM,
        prompt_template: str,
        max_contexts: int = 5,
        context_window: int = 8000
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Document retriever
            llm: Language model for generation
            prompt_template: Prompt template with {context} and {question} placeholders
            max_contexts: Maximum number of contexts to use
            context_window: Maximum context window size
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_contexts = max_contexts
        self.context_window = context_window
        
        logger.info("Initialized RAG pipeline")
    
    def generate_answer(
        self,
        query_id: str,
        query_text: str,
        retrieval_result: RetrievalResult
    ) -> RAGResponse:
        """
        Generate answer for a query using retrieved context
        
        Args:
            query_id: Query ID
            query_text: Query text
            retrieval_result: Retrieved documents
            
        Returns:
            RAGResponse object
        """
        # Format context
        context_str = self.retriever.format_context(
            retrieval_result,
            max_contexts=self.max_contexts
        )
        
        # Truncate context if too long (simple word-based truncation)
        words = context_str.split()
        if len(words) > self.context_window:
            context_str = " ".join(words[:self.context_window])
            logger.warning(f"Truncated context for query {query_id}")
        
        # Create prompt
        prompt = self.prompt_template.format(
            context=context_str,
            question=query_text
        )
        
        # Generate answer
        answer = self.llm.generate(prompt)
        
        # Get all retrieved documents for evaluation
        all_docs_with_scores = self.retriever.get_retrieved_documents(
            retrieval_result,
            include_scores=True
        )
        
        # Use only max_contexts for answer generation
        contexts = [doc.text for doc, _ in all_docs_with_scores[:self.max_contexts]]
        
        # But save all retrieved doc IDs for proper evaluation
        # This ensures recall@k metrics are calculated correctly for all k values
        context_ids = [doc.id for doc, _ in all_docs_with_scores]
        scores = [score for _, score in all_docs_with_scores]
        
        return RAGResponse(
            query_id=query_id,
            query_text=query_text,
            answer=answer,
            contexts=contexts,
            context_ids=context_ids,
            retrieval_scores=scores,
            metadata={
                'context_length': len(context_str.split()),
                'num_contexts_used': len(contexts),  # Used for answer generation
                'num_contexts_retrieved': len(context_ids)  # Available for evaluation
            }
        )
    
    def batch_generate(
        self,
        queries: Dict,
        retrieval_results: Dict[str, RetrievalResult],
        show_progress: bool = True
    ) -> Dict[str, RAGResponse]:
        """
        Generate answers for multiple queries
        
        Args:
            queries: Dictionary of Query objects
            retrieval_results: Dictionary of retrieval results
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping query IDs to RAGResponse objects
        """
        responses = {}
        
        iterator = tqdm(queries.items()) if show_progress else queries.items()
        
        for query_id, query in iterator:
            if query_id not in retrieval_results:
                logger.warning(f"No retrieval results for query {query_id}")
                continue
            
            try:
                response = self.generate_answer(
                    query_id=query_id,
                    query_text=query.text,
                    retrieval_result=retrieval_results[query_id]
                )
                responses[query_id] = response
            except Exception as e:
                logger.error(f"Error generating answer for query {query_id}: {e}")
                continue
        
        logger.info(f"Generated answers for {len(responses)} queries")
        return responses
    
    def run_pipeline(
        self,
        queries: Dict,
        query_embeddings: Dict,
        show_progress: bool = True
    ) -> Dict[str, RAGResponse]:
        """
        Run complete RAG pipeline: retrieval + generation
        
        Args:
            queries: Dictionary of Query objects
            query_embeddings: Dictionary of query embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping query IDs to RAGResponse objects
        """
        # Step 1: Retrieve documents
        logger.info("Step 1: Retrieving documents...")
        retrieval_results = self.retriever.batch_retrieve(
            queries=queries,
            query_embeddings=query_embeddings
        )
        
        # Step 2: Generate answers
        logger.info("Step 2: Generating answers...")
        responses = self.batch_generate(
            queries=queries,
            retrieval_results=retrieval_results,
            show_progress=show_progress
        )
        
        return responses


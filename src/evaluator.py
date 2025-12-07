"""
Evaluation system for FinRAG using Ragas framework
"""

import pandas as pd
from typing import Dict, List
from datasets import Dataset
import logging

from .rag_pipeline import RAGResponse

logger = logging.getLogger(__name__)

# Try to import Ragas (it may not be installed in baseline)
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    # Ragas 0.4.0+ renamed context_relevancy to context_recall
    # Using available metrics
    RAGAS_AVAILABLE = True
    logger.info("Ragas imported successfully")
except ImportError as e:
    logger.warning(f"Ragas not available. Install with: pip install ragas. Error: {e}")
    RAGAS_AVAILABLE = False

# Import Gemini for Ragas evaluation
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
    logger.info("Gemini available for Ragas evaluation")
except ImportError:
    logger.warning("langchain_google_genai not available. Install with: pip install langchain-google-genai")
    GEMINI_AVAILABLE = False


class RAGEvaluator:
    """Evaluator for RAG system"""
    
    def __init__(
        self,
        use_ragas: bool = True,
        metrics: List[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            use_ragas: Whether to use Ragas for evaluation
            metrics: List of metrics to compute
        """
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        self.llm = None
        
        if metrics is None:
            self.metrics = [
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_relevancy"
            ]
        else:
            self.metrics = metrics
        
        if self.use_ragas:
            # Map metric names to Ragas metric objects
            self.ragas_metrics = []
            metric_map = {
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
            }
            
            for metric_name in self.metrics:
                if metric_name in metric_map:
                    self.ragas_metrics.append(metric_map[metric_name])
            
            # Initialize Gemini LLM for Ragas evaluation
            if GEMINI_AVAILABLE:
                try:
                    import os
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY not found in environment")
                    
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        google_api_key=api_key,
                        temperature=0.1,
                        max_output_tokens=2048
                    )
                    logger.info("Initialized Ragas with Gemini (gemini-pro)")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini: {e}")
                    logger.error("Please set GOOGLE_API_KEY in .env file")
                    raise
            else:
                logger.error("Gemini not available. Install with: pip install langchain-google-genai")
                raise ImportError("langchain_google_genai is required for Ragas evaluation")
            
            logger.info(f"Initialized Ragas evaluator with {len(self.ragas_metrics)} metrics")
        else:
            logger.info("Initialized basic evaluator (Ragas not available)")
    
    def prepare_ragas_dataset(
        self,
        responses: Dict[str, RAGResponse],
        ground_truths: Dict[str, str] = None
    ) -> Dataset:
        """
        Prepare dataset for Ragas evaluation
        
        Args:
            responses: Dictionary of RAG responses
            ground_truths: Dictionary of ground truth answers (optional)
            
        Returns:
            HuggingFace Dataset
        """
        data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for query_id, response in responses.items():
            data['question'].append(response.query_text)
            data['answer'].append(response.answer)
            data['contexts'].append(response.contexts)
            
            # Add ground truth if available
            if ground_truths and query_id in ground_truths:
                data['ground_truth'].append(ground_truths[query_id])
            else:
                data['ground_truth'].append("")  # Ragas can work without ground truth for some metrics
        
        return Dataset.from_dict(data)
    
    def evaluate_with_ragas(
        self,
        responses: Dict[str, RAGResponse],
        ground_truths: Dict[str, str] = None
    ) -> Dict:
        """
        Evaluate using Ragas framework
        
        Args:
            responses: Dictionary of RAG responses
            ground_truths: Dictionary of ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.use_ragas:
            raise RuntimeError("Ragas is not available")
        
        # Prepare dataset
        dataset = self.prepare_ragas_dataset(responses, ground_truths)
        
        logger.info(f"Evaluating {len(dataset)} responses with Ragas...")
        if self.llm:
            logger.info("Using AWS Bedrock LLM for Ragas evaluation")
        else:
            logger.info("Using default LLM (OpenAI) for Ragas evaluation")
        
        # Run evaluation
        try:
            # Pass LLM to Ragas if available
            if self.llm:
                results = evaluate(
                    dataset,
                    metrics=self.ragas_metrics,
                    llm=self.llm
                )
            else:
                results = evaluate(
                    dataset,
                    metrics=self.ragas_metrics
                )
            
            # Convert to dictionary
            metrics_dict = {
                metric: float(value) 
                for metric, value in results.items()
            }
            
            logger.info("Ragas evaluation completed")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {}
    
    def evaluate_retrieval_quality(
        self,
        responses: Dict[str, RAGResponse],
        qrels: Dict[str, List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval quality using traditional IR metrics
        
        Args:
            responses: Dictionary of RAG responses
            qrels: Ground truth relevance judgments
            k_values: K values for metrics
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {f"recall@{k}": [] for k in k_values}
        metrics.update({f"precision@{k}": [] for k in k_values})
        metrics["mrr"] = []
        
        for query_id, response in responses.items():
            if query_id not in qrels:
                continue
            
            relevant_docs = set(qrels[query_id])
            retrieved_ids = response.context_ids
            
            # Calculate metrics
            for k in k_values:
                top_k = retrieved_ids[:k]
                relevant_in_k = len(set(top_k) & relevant_docs)
                
                recall = relevant_in_k / len(relevant_docs) if relevant_docs else 0
                metrics[f"recall@{k}"].append(recall)
                
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
        averaged = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in metrics.items()
        }
        
        return averaged
    
    def evaluate_all(
        self,
        responses: Dict[str, RAGResponse],
        qrels: Dict[str, List[str]] = None,
        ground_truths: Dict[str, str] = None
    ) -> Dict:
        """
        Run all available evaluations
        
        Args:
            responses: Dictionary of RAG responses
            qrels: Ground truth relevance judgments for retrieval
            ground_truths: Ground truth answers for generation
            
        Returns:
            Dictionary of all evaluation metrics
        """
        all_metrics = {}
        
        # Retrieval metrics
        if qrels:
            logger.info("Evaluating retrieval quality...")
            retrieval_metrics = self.evaluate_retrieval_quality(responses, qrels)
            all_metrics['retrieval'] = retrieval_metrics
        
        # Ragas metrics
        if self.use_ragas:
            logger.info("Evaluating with Ragas...")
            ragas_metrics = self.evaluate_with_ragas(responses, ground_truths)
            all_metrics['ragas'] = ragas_metrics
        
        return all_metrics
    
    def generate_report(
        self,
        metrics: Dict,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Generate evaluation report
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_path: Path to save report (optional)
            
        Returns:
            DataFrame with metrics
        """
        # Flatten nested metrics
        flat_metrics = {}
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for metric_name, value in category_metrics.items():
                    flat_metrics[f"{category}/{metric_name}"] = value
            else:
                flat_metrics[category] = category_metrics
        
        # Create DataFrame
        df = pd.DataFrame([flat_metrics])
        
        # Save if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved evaluation report to {output_path}")
        
        return df
    
    def compare_results(
        self,
        baseline_metrics: Dict,
        current_metrics: Dict
    ) -> pd.DataFrame:
        """
        Compare baseline and current metrics
        
        Args:
            baseline_metrics: Baseline metrics
            current_metrics: Current metrics
            
        Returns:
            DataFrame with comparison
        """
        # Flatten metrics
        def flatten(d, prefix=''):
            items = []
            for k, v in d.items():
                new_key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key))
                else:
                    items.append((new_key, v))
            return items
        
        baseline_flat = dict(flatten(baseline_metrics))
        current_flat = dict(flatten(current_metrics))
        
        # Create comparison
        comparison = []
        for metric in set(baseline_flat.keys()) | set(current_flat.keys()):
            baseline_val = baseline_flat.get(metric, 0)
            current_val = current_flat.get(metric, 0)
            improvement = current_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
            
            comparison.append({
                'Metric': metric,
                'Baseline': baseline_val,
                'Current': current_val,
                'Improvement': improvement,
                'Improvement (%)': improvement_pct
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Improvement (%)', ascending=False)
        
        return df


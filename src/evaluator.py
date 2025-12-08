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

# Import LLM options for Ragas evaluation
try:
    from langchain_aws import ChatBedrock
    BEDROCK_LLM_AVAILABLE = True
except ImportError:
    BEDROCK_LLM_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
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
            
            # Initialize Gemini for Ragas evaluation
            # Use Gemini 2.0 Flash for better JSON parsing and objective evaluation
            if GEMINI_AVAILABLE:
                try:
                    import os
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY not found in .env file")
                    
                    # Use Gemini 2.5 Flash (latest, best JSON support for Ragas)
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key,
                        temperature=0.1,
                        max_output_tokens=2048
                    )
                    logger.info("Initialized Ragas with Gemini 2.5 Flash")
                    logger.info("Using Gemini for objective evaluation (avoids self-evaluation bias)")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini: {e}")
                    logger.error("Please ensure GOOGLE_API_KEY is set in .env file")
                    logger.error("Get your API key from: https://aistudio.google.com/apikey")
                    raise
            else:
                logger.error("Gemini not available. Install with: pip install langchain-google-genai")
                raise ImportError("langchain-google-genai is required for Ragas evaluation")
            
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
            logger.info("Using Google Gemini for Ragas evaluation")
        else:
            logger.error("Gemini LLM not configured! Check GOOGLE_API_KEY in .env")
            raise RuntimeError("Gemini LLM is required for Ragas evaluation")
        
        # Run evaluation
        try:
            import os
            
            # Workaround: Set dummy OPENAI_API_KEY to avoid Ragas internal errors
            # Ragas may check for this even when using other LLMs
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "sk-dummy-key-not-used"
                logger.debug("Set placeholder OPENAI_API_KEY (Bedrock will be used)")
            
            # Configure Gemini embeddings for Ragas (some metrics need it)
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            import os
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Pass both LLM and embeddings to Ragas
            logger.info("Starting Ragas evaluation (this may take several minutes)...")
            logger.info(f"Evaluating {len(dataset)} samples with {len(self.ragas_metrics)} metrics")
            logger.warning("Ragas evaluation is slow - each sample may take 2-4 minutes")
            
            # Note: Ragas 0.4.0 evaluate() supports: dataset, metrics, llm, embeddings, raise_exceptions
            # It does NOT support: max_workers, show_progress
            results = evaluate(
                dataset,
                metrics=self.ragas_metrics,
                llm=self.llm,
                embeddings=embeddings,
                raise_exceptions=False  # Don't fail on individual metric errors
            )
            
            # Convert to dictionary (Ragas 0.4 returns EvaluationResult object)
            metrics_dict = {}
            
            try:
                # Method 1: Try to_pandas() (most common for Ragas 0.4+)
                if hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    for col in df.columns:
                        if col in self.metrics:
                            # Calculate mean, ignoring NaN values
                            values = df[col].dropna()
                            if len(values) > 0:
                                metrics_dict[col] = float(values.mean())
                            else:
                                logger.warning(f"Metric {col} has no valid values")
                    logger.info(f"Extracted {len(metrics_dict)} metrics from Ragas results")
                
                # Method 2: Try direct dictionary access
                elif isinstance(results, dict):
                    metrics_dict = {k: float(v) for k, v in results.items() if k in self.metrics}
                
                # Method 3: Try scores attribute
                elif hasattr(results, 'scores'):
                    for metric in self.metrics:
                        if metric in results.scores:
                            metrics_dict[metric] = float(results.scores[metric])
                
                else:
                    logger.warning(f"Unknown Ragas result type: {type(results)}")
                    logger.warning(f"Result attributes: {dir(results)}")
                    
            except Exception as e:
                logger.error(f"Failed to extract metrics from Ragas results: {e}")
            
            if metrics_dict:
                logger.info("Ragas evaluation completed successfully")
                return metrics_dict
            else:
                logger.warning("Ragas evaluation completed but returned no valid metrics")
                return {}
            
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
                # Ensure we have enough retrieved documents for this k value
                if len(retrieved_ids) < k:
                    logger.warning(
                        f"Query {query_id}: only {len(retrieved_ids)} docs retrieved, "
                        f"but evaluating @{k} metrics. Using all available docs."
                    )
                
                top_k = retrieved_ids[:k]  # Will use all if len < k
                relevant_in_k = len(set(top_k) & relevant_docs)
                
                recall = relevant_in_k / len(relevant_docs) if relevant_docs else 0
                metrics[f"recall@{k}"].append(recall)
                
                # For precision, use actual number of docs retrieved
                actual_k = len(top_k)
                precision = relevant_in_k / actual_k if actual_k > 0 else 0
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


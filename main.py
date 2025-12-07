"""
Main execution script for FinRAG baseline system
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from dotenv import load_dotenv
import time
import numpy as np

# Fix OpenMP conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables from .env file
load_dotenv()

from src.config import load_config
from src.data_loader import DataLoader, merge_corpora, merge_queries
from src.embeddings import BedrockEmbeddings
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever, evaluate_retrieval
from src.rag_pipeline import BedrockLLM, RAGPipeline
from src.evaluator import RAGEvaluator


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def main(args):
    """Main execution function"""
    
    # Setup basic logging first
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Update logging with config settings
    if not args.no_log:
        setup_logging(
            log_file=os.path.join(config.data.output_dir, "finrag.log"),
            level=args.log_level
        )
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FinRAG Baseline System")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 80)
    
    data_loader = DataLoader(config.data.base_path)
    
    # Load all datasets
    dataset_configs = [
        {
            'name': ds.name,
            'corpus': ds.corpus,
            'queries': ds.queries,
            'qrels': ds.qrels
        }
        for ds in config.data.datasets
    ]
    
    # Load specific dataset if specified
    if args.dataset:
        dataset_configs = [dc for dc in dataset_configs if dc['name'] == args.dataset]
        if not dataset_configs:
            raise ValueError(f"Dataset '{args.dataset}' not found in config")
    
    datasets = data_loader.load_all_datasets(dataset_configs)
    
    # Merge all corpora and queries
    all_documents = merge_corpora(datasets)
    all_queries = merge_queries(datasets)
    
    # Merge qrels
    all_qrels = {}
    for dataset_name, (_, _, qrels) in datasets.items():
        all_qrels.update(qrels)
    
    logger.info(f"Total documents: {len(all_documents)}")
    logger.info(f"Total queries: {len(all_queries)}")
    logger.info(f"Total qrels: {len(all_qrels)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Generate Embeddings or Load from Cache
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Embedding Generation")
    logger.info("=" * 80)
    
    # Initialize embeddings
    embedder = BedrockEmbeddings(
        model_id=config.aws.embedding_model,
        region=config.aws.region,
        dimension=config.vector_store.dimension,
        batch_size=config.performance.batch_size,
        cache_dir=str(cache_dir) if config.performance.cache_embeddings else None
    )
    
    # Generate document embeddings
    if not args.skip_embeddings:
        logger.info("Generating document embeddings...")
        doc_embeddings = embedder.embed_documents(all_documents)
        
        logger.info("Generating query embeddings...")
        query_embeddings = embedder.embed_queries(all_queries)
    else:
        logger.info("Skipping embedding generation (using cached embeddings)")
        doc_embeddings = embedder.embed_documents(all_documents)
        query_embeddings = embedder.embed_queries(all_queries)
    
    # -------------------------------------------------------------------------
    # Step 3: Build or Load Vector Store
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Vector Store")
    logger.info("=" * 80)
    
    index_path = Path(config.vector_store.index_path)
    
    if args.rebuild_index or not index_path.exists():
        logger.info("Building vector store...")
        vector_store = FAISSVectorStore(
            dimension=config.vector_store.dimension,
            metric=config.vector_store.metric,
            index_path=str(index_path)
        )
        
        vector_store.add_documents(doc_embeddings)
        vector_store.save()
        logger.info(f"Vector store saved to {index_path}")
    else:
        logger.info(f"Loading vector store from {index_path}")
        vector_store = FAISSVectorStore.load(str(index_path))
    
    logger.info(f"Vector store stats: {vector_store.get_stats()}")
    
    # -------------------------------------------------------------------------
    # Step 4: Retrieval
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Document Retrieval")
    logger.info("=" * 80)
    
    retriever = Retriever(
        vector_store=vector_store,
        documents=all_documents,
        top_k=config.retrieval.top_k
    )
    
    # Retrieve documents for all queries
    logger.info(f"Retrieving documents for {len(all_queries)} queries...")
    retrieval_start = time.time()
    
    retrieval_results = retriever.batch_retrieve(
        queries=all_queries,
        query_embeddings=query_embeddings,
        top_k=config.retrieval.top_k
    )
    
    retrieval_time = time.time() - retrieval_start
    avg_retrieval_latency = retrieval_time / len(all_queries)
    logger.info(f"Retrieval completed in {retrieval_time:.2f}s")
    logger.info(f"Average retrieval latency: {avg_retrieval_latency:.3f}s per query")
    
    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(
        retrieval_results=retrieval_results,
        qrels=all_qrels,
        k_values=[1, 5, 10]
    )
    
    logger.info("Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save retrieval results
    retrieval_output = output_dir / f"retrieval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(retrieval_output, 'w') as f:
        results_data = {
            qid: {
                'query': result.query_text,
                'retrieved': [
                    {'doc_id': doc_id, 'score': float(score)}
                    for doc_id, score in result.retrieved_docs
                ]
            }
            for qid, result in retrieval_results.items()
        }
        json.dump(results_data, f, indent=2)
    logger.info(f"Retrieval results saved to {retrieval_output}")
    
    # -------------------------------------------------------------------------
    # Step 5: RAG Pipeline (Optional - can be expensive)
    # -------------------------------------------------------------------------
    if not args.skip_generation:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Answer Generation")
        logger.info("=" * 80)
        
        # Initialize LLM
        llm = BedrockLLM(
            model_id=config.aws.llm_model,
            region=config.aws.region,
            max_tokens=config.aws.max_tokens,
            temperature=config.aws.temperature,
            top_p=config.aws.top_p
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            prompt_template=config.rag.prompt_template,
            max_contexts=config.rag.max_contexts,
            context_window=config.rag.context_window
        )
        
        # Limit queries if specified
        if args.max_queries:
            query_subset = dict(list(all_queries.items())[:args.max_queries])
            logger.info(f"Processing {len(query_subset)} queries (limited)")
        else:
            query_subset = all_queries
        
        # Generate answers
        logger.info(f"Generating answers for {len(query_subset)} queries...")
        generation_start = time.time()
        
        rag_responses = rag_pipeline.run_pipeline(
            queries=query_subset,
            query_embeddings=query_embeddings,
            show_progress=True
        )
        
        generation_time = time.time() - generation_start
        avg_generation_latency = generation_time / len(query_subset)
        logger.info(f"Answer generation completed in {generation_time:.2f}s")
        logger.info(f"Average generation latency: {avg_generation_latency:.3f}s per query")
        
        # Save RAG responses
        rag_output = output_dir / f"rag_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(rag_output, 'w') as f:
            responses_data = {
                qid: {
                    'query': resp.query_text,
                    'answer': resp.answer,
                    'contexts': resp.contexts,
                    'context_ids': resp.context_ids,
                    'scores': resp.retrieval_scores,
                    'metadata': resp.metadata
                }
                for qid, resp in rag_responses.items()
            }
            json.dump(responses_data, f, indent=2)
        logger.info(f"RAG responses saved to {rag_output}")
        
        # -------------------------------------------------------------------------
        # Step 6: Evaluation
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Evaluation")
        logger.info("=" * 80)
        
        evaluator = RAGEvaluator(
            use_ragas=args.use_ragas,
            metrics=config.evaluation.metrics
        )
        
        # Run evaluation
        all_metrics = evaluator.evaluate_all(
            responses=rag_responses,
            qrels=all_qrels
        )
        
        # Generate report
        report = evaluator.generate_report(
            metrics=all_metrics,
            output_path=str(output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        )
        
        logger.info("\nEvaluation Results:")
        logger.info(report.to_string())
        
    else:
        logger.info("\nSkipping answer generation (use --generate to enable)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)
    
    # Performance Summary
    logger.info("\n📊 Performance Summary:")
    logger.info(f"  Total queries processed: {len(all_queries)}")
    logger.info(f"  Average retrieval latency: {avg_retrieval_latency:.3f}s per query")
    
    if not args.skip_generation:
        logger.info(f"  Average generation latency: {avg_generation_latency:.3f}s per query")
        total_avg_latency = avg_retrieval_latency + avg_generation_latency
        logger.info(f"  Average total latency: {total_avg_latency:.3f}s per query")
        
        # Cost estimation
        estimated_cost_per_query = 0.015  # Rough estimate for Titan Text Express
        total_cost = estimated_cost_per_query * len(query_subset)
        logger.info(f"\n💰 Cost Estimation (approximate):")
        logger.info(f"  Cost per query: ~${estimated_cost_per_query:.4f}")
        logger.info(f"  Total cost: ~${total_cost:.2f}")
    
    logger.info(f"\n📁 Output directory: {output_dir}")
    logger.info("FinRAG baseline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinRAG Baseline System")
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to process (default: all)"
    )
    
    # Processing options
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use cached)"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild vector index even if exists"
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer generation (retrieval only)"
    )
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process"
    )
    
    # Evaluation options
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="Use Ragas for evaluation (requires OpenAI API key)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file output"
    )
    
    args = parser.parse_args()
    
    main(args)


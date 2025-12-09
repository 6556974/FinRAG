"""
Interactive Demo Script for FinRAG Baseline System
Run this script to see a step-by-step demonstration
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import time
import numpy as np

# Fix OpenMP conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.data_loader import DataLoader, merge_corpora, Query
from src.embeddings import BedrockEmbeddings
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever, evaluate_retrieval
from src.rag_pipeline import BedrockLLM, GeminiLLM, RAGPipeline


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo():
    """Run interactive demo"""
    
    print_section("FinRAG Baseline Demo")
    
    # 1. Load configuration
    print_section("Step 1: Loading Configuration")
    config = load_config("config.yaml")
    logger.info(f"Embedding Model: {config.aws.embedding_model}")
    if config.llm_provider == "gemini":
        logger.info(f"LLM Model: {config.gemini.model_id}")
        llm = GeminiLLM(
            model_id=config.gemini.model_id,
            max_tokens=config.gemini.max_tokens,
            temperature=config.gemini.temperature,
            top_p=config.gemini.top_p
        )
    else:
        logger.info(f"LLM Model: {config.aws.llm_model}")
        llm = BedrockLLM(
            model_id=config.aws.llm_model,
            region=config.aws.region,
            max_tokens=config.aws.max_tokens,
            temperature=config.aws.temperature,
            top_p=config.aws.top_p
        )
    
    # 2. Load data
    print_section("Step 2: Loading All Datasets")
    data_loader = DataLoader(config.data.base_path)
    
    # Load all three datasets
    dataset_configs = [
        {
            'name': ds.name,
            'corpus': ds.corpus,
            'queries': ds.queries,
            'qrels': ds.qrels
        }
        for ds in config.data.datasets
    ]
    
    datasets = data_loader.load_all_datasets(dataset_configs)
    documents = merge_corpora(datasets)
    
    logger.info(f"Loaded {len(documents)} documents from {len(datasets)} datasets")
    logger.info(f"Datasets: {', '.join(datasets.keys())}")
    
    # 3. Generate embeddings
    print_section("Step 3: Generating Embeddings")
    embedder = BedrockEmbeddings(
        model_id=config.aws.embedding_model,
        region=config.aws.region,
        dimension=config.vector_store.dimension,
        cache_dir="cache"
    )
    
    logger.info("Generating document embeddings...")
    doc_embeddings = embedder.embed_documents(documents)
    
    # 4. Build vector store
    print_section("Step 4: Building Vector Store")
    vector_store = FAISSVectorStore(
        dimension=config.vector_store.dimension,
        metric="cosine"
    )
    vector_store.add_documents(doc_embeddings)
    logger.info(f"Built vector store with {vector_store.index.ntotal} documents")
    
    # 5. Initialize Retriever and RAG Pipeline
    print_section("Step 5: Initializing RAG Pipeline")
    
    retriever = Retriever(
        vector_store=vector_store,
        documents=documents,
        top_k=10
    )
    
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_template=config.rag.prompt_template,
        max_contexts=config.rag.max_contexts
    )
    
    logger.info("✓ RAG pipeline ready")
    
    # 6. Test Custom Questions (Phase 1 Request Set)
    print_section("Step 6: Testing Phase 1 Request Set (10 Questions)")
    
    def ask_custom_question(question: str):
        """Ask a custom question"""
        query_embedding = embedder._get_embedding(question)
        
        result = retriever.retrieve(
            query_id="custom",
            query_text=question,
            query_embedding=query_embedding
        )
        
        response = rag.generate_answer(
            query_id="custom",
            query_text=question,
            retrieval_result=result
        )
        
        return response, result
    
    # 10 Custom test questions
    example_questions = [
        "How much revenue does Microsoft generate from contracts with customers?",
        "When did Coupang's Farfetch consolidation start?",
        "What was the change in total expense net of tax for share based compensation from 2014 to 2015 in millions?",
        "Did abiomed outperform the nasdaq medical equipment index?",
        "How much revenue does Microsoft generate from contracts with customers?",
        "When did Coupang's Farfetch consolidation start?",
        "What is CPNG's free cash flow?",
        "What was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?",
        "what is the growth rate in the balance of standby letters of credit from 2006 to 2007?",
        "what is the percentage change in revenue generated from non-us currencies from 2015 to 2016?"
    ]
    
    print(f"Testing {len(example_questions)} custom questions:\n")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q[:80]}..." if len(q) > 80 else f"{i}. {q}")
    
    # Test all 10 questions
    print("\n" + "="*70)
    print("Testing all 10 questions...")
    print("="*70)
    
    all_responses = []
    query_latencies = []
    test_start = time.time()
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/10")
        print(f"{'='*70}")
        print(f"Q: {question}\n")
        
        try:
            query_start = time.time()
            response, result = ask_custom_question(question)
            query_time = time.time() - query_start
            query_latencies.append(query_time)
            
            print(f"Latency: {query_time:.2f}s")
            print(f"\nA: {response.answer[:200]}..." if len(response.answer) > 200 else f"\nA: {response.answer}")
            all_responses.append((question, response.answer, True))
        except Exception as e:
            print(f"❌ Error: {e}")
            all_responses.append((question, str(e), False))
    
    successful = sum(1 for _, _, success in all_responses if success)
    
    # Summary
    print_section("Summary")
    print(f"Tested: {len(example_questions)} questions")
    print(f"Successful: {successful}/{len(example_questions)}")
    
    if query_latencies:
        avg_latency = np.mean(query_latencies)
        print(f"\nAverage latency: {avg_latency:.2f}s per query")
    
    # Done
    print_section("Demo Complete!")


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


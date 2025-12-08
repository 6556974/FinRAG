"""
Interactive Q&A Script for FinRAG
Use this to ask your own questions to the financial documents
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.data_loader import DataLoader, Query, merge_corpora
from src.embeddings import BedrockEmbeddings
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.rag_pipeline import BedrockLLM, GeminiLLM, RAGPipeline


def ask_question(question: str, dataset: str = None, top_k: int = 5, show_contexts: bool = True):
    """
    Ask a custom question to the FinRAG system
    
    Args:
        question: Your custom question
        dataset: Which dataset to search in (default: None = search all datasets)
        top_k: Number of documents to retrieve
        show_contexts: Whether to show retrieved contexts
        
    Returns:
        Answer string
    """
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"{'='*70}\n")
    
    # Load configuration
    config = load_config("config.yaml")
    data_loader = DataLoader(config.data.base_path)
    if config.llm_provider == "gemini":
        llm = GeminiLLM(
            model_id=config.gemini.model_id,
            max_tokens=config.gemini.max_tokens,
            temperature=config.gemini.temperature,
            top_p=config.gemini.top_p
        )
    else:
        llm = BedrockLLM(
            model_id=config.aws.llm_model,
            region=config.aws.region,
            max_tokens=config.aws.max_tokens,
            temperature=config.aws.temperature,
            top_p=config.aws.top_p
        )
    
    # Load datasets
    if dataset:
        # Search specific dataset
        print(f"Loading {dataset} dataset...")
        dataset_config = [ds for ds in config.data.datasets if ds.name == dataset]
        if not dataset_config:
            print(f"Error: Dataset '{dataset}' not found in config.yaml")
            print(f"Available datasets: {[ds.name for ds in config.data.datasets]}")
            return None
        
        ds = dataset_config[0]
        documents, _, _ = data_loader.load_dataset(
            corpus_file=ds.corpus,
            queries_file=ds.queries,
            qrels_file=ds.qrels
        )
        print(f"✓ Loaded {len(documents)} documents from {dataset}")
    else:
        # Search all datasets (default)
        print(f"Loading all datasets...")
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
        print(f"✓ Loaded {len(documents)} documents from {len(datasets)} datasets")
        print(f"  Datasets: {', '.join(datasets.keys())}")
    
    # Initialize embeddings
    embedder = BedrockEmbeddings(
        model_id=config.aws.embedding_model,
        region=config.aws.region,
        dimension=config.vector_store.dimension,
        cache_dir=config.data.cache_dir
    )
    
    # Check if index exists
    index_path = Path(config.vector_store.index_path)
    
    if index_path.exists():
        print(f"✓ Loading existing vector index...")
        vector_store = FAISSVectorStore.load(str(index_path))
    else:
        print(f"Building vector index (first time only)...")
        doc_embeddings = embedder.embed_documents(documents)
        vector_store = FAISSVectorStore(
            dimension=config.vector_store.dimension,
            metric=config.vector_store.metric,
            index_path=str(index_path)
        )
        vector_store.add_documents(doc_embeddings)
        vector_store.save()
        print(f"✓ Vector index saved")
    
    # Create retriever
    retriever = Retriever(
        vector_store=vector_store,
        documents=documents,
        top_k=top_k
    )
    
    # Generate query embedding
    print(f"Searching for relevant documents...")
    query_embedding = embedder._get_embedding(question)
    
    # Retrieve documents
    retrieval_result = retriever.retrieve(
        query_id="custom",
        query_text=question,
        query_embedding=query_embedding
    )
    
    # Show retrieved contexts
    if show_contexts:
        print(f"\n✓ Found {len(retrieval_result.retrieved_docs)} relevant documents:\n")
        for i, (doc_id, score) in enumerate(retrieval_result.retrieved_docs[:3], 1):
            doc = documents[doc_id]
            print(f"{i}. [Score: {score:.4f}] {doc.title}")
            print(f"   {doc.text[:150]}...\n")
    
    # Generate answer
    print("Generating answer...")
    
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_template=config.rag.prompt_template,
        max_contexts=config.rag.max_contexts
    )
    
    response = rag.generate_answer(
        query_id="custom",
        query_text=question,
        retrieval_result=retrieval_result
    )
    
    # Display answer
    print(f"\n{'='*70}")
    print("ANSWER:")
    print(f"{'='*70}")
    print(f"{response.answer}")
    print(f"{'='*70}\n")
    
    return response.answer


def main():
    """Main function for custom queries"""
    
    print("\n" + "="*70)
    print("  FinRAG Interactive Q&A")
    print("="*70)
    print("\nAsk questions about financial documents!")
    print("Searching across ALL datasets by default (32K+ documents)")
    print("Use --dataset to search specific dataset (financebench, finqa, finder, etc.)")
    print("Press Ctrl+C to exit\n")
    
    # Example usage
    example_questions = [
        "What is Boeing's total revenue in 2022?",
        "Does Paypal have positive working capital based on 2022 data?",
        "What is PepsiCo's operating margin?",
        "How much did Adobe spend on research and development?",
        "What is Microsoft's revenue growth rate?"
    ]
    
    print("Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-"*70)
    
    # Interactive mode
    try:
        while True:
            print("\nEnter your question (or 'quit' to exit):")
            user_question = input("> ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_question:
                continue
            
            # Ask the question
            answer = ask_question(
                question=user_question,
                dataset="financebench",  # Change to finqa or finder if needed
                top_k=5,
                show_contexts=True
            )
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ask custom questions to FinRAG")
    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("--dataset", default=None, help="Dataset to search (default: all datasets)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--no-contexts", action="store_true", help="Don't show contexts")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question mode
        ask_question(
            question=args.question,
            dataset=args.dataset,
            top_k=args.top_k,
            show_contexts=not args.no_contexts
        )
    else:
        # Interactive mode
        main()


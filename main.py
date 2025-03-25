"""
Main entry point for running the RAG system.

This script provides a command-line interface for the RAG system.

Usage:
    # Run the web interface
    python main.py web
    
    # Process documents and add them to the database
    python main.py ingest <file_or_directory_path>
    
    # Query the system
    python main.py query "Your question here"
    
    # Clear the vector database
    python main.py clear
"""

import os
import sys
import argparse
from pathlib import Path

from src import config
from src.rag_pipeline import RAGPipeline
from src import vector_store


def run_web():
    """Run the web interface."""
    try:
        import streamlit
        os.system("streamlit run src/webui.py --client.showErrorDetails=false --server.fileWatcherType=none")
    except ImportError:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        sys.exit(1)


def ingest_documents(path):
    """
    Ingest documents from a file or directory.
    
    Args:
        path: Path to a file or directory
    """
    path = Path(path)
    
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline()
    
    if path.is_file():
        files = [str(path)]
    else:
        # Find all files in directory
        files = [str(f) for f in path.glob("**/*") if f.is_file()]
        print(f"Found {len(files)} files in {path}")
    
    # Process files
    if pipeline.process_files(files):
        doc_count = pipeline.get_document_count()
        print(f"Successfully processed {len(files)} files. Vector store now contains {doc_count} chunks.")
    else:
        print("Failed to process files")
        sys.exit(1)


def query(question):
    """
    Query the RAG system.
    
    Args:
        question: Question to ask
    """
    # Initialize the RAG pipeline
    pipeline = RAGPipeline()
    
    if pipeline.get_document_count() == 0:
        print("No documents have been ingested yet. Please ingest documents first.")
        sys.exit(1)
    
    print(f"Question: {question}")
    print("Generating answer...")
    
    result = pipeline.query(question)
    
    print("\nAnswer:")
    print(result["answer"])
    print(f"\nQuery processed in {result['query_time']:.2f} seconds")
    
    # Ask if user wants to see sources
    show_sources = input("\nShow sources? (y/n): ")
    if show_sources.lower() == "y":
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\nSource {i}:")
            print(source[:500] + "..." if len(source) > 500 else source)
            print("-" * 80)


def clear_db():
    """Clear the vector database."""
    if vector_store.clear_vector_store():
        print("Vector database cleared successfully")
    else:
        print("Failed to clear vector database")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Run the web interface")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to a file or directory")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the vector database")
    
    args = parser.parse_args()
    
    if args.command == "web":
        run_web()
    elif args.command == "ingest":
        ingest_documents(args.path)
    elif args.command == "query":
        query(args.question)
    elif args.command == "clear":
        clear_db()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
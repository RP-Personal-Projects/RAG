"""
Module for managing vector store operations.
"""

from typing import List, Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

from . import config
from . import embeddings


def get_vector_store(embedding_function: Optional[Embeddings] = None):
    """
    Get the vector store.
    
    Args:
        embedding_function: Embedding function to use
        
    Returns:
        Vector store or None if it doesn't exist
    """
    if embedding_function is None:
        embedding_function = embeddings.get_embedding_model()
    
    if not config.PERSIST_DIRECTORY or not config.os.path.exists(config.PERSIST_DIRECTORY):
        config.logger.warning(f"Vector store directory does not exist: {config.PERSIST_DIRECTORY}")
        return None
    
    try:
        config.logger.info(f"Loading vector store from: {config.PERSIST_DIRECTORY}")
        return Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )
    except Exception as e:
        config.logger.error(f"Error loading vector store: {e}")
        return None


def create_or_update_vector_store(documents: List[Document], embedding_function: Optional[Embeddings] = None):
    """
    Create or update the vector store with documents.
    
    Args:
        documents: Documents to add to the vector store
        embedding_function: Embedding function to use
        
    Returns:
        Updated vector store
    """
    if not documents:
        config.logger.warning("No documents to add to vector store")
        return None
    
    if embedding_function is None:
        embedding_function = embeddings.get_embedding_model()
    
    # Get existing vector store or create new one
    existing_store = get_vector_store(embedding_function)
    
    if existing_store is None:
        config.logger.info(f"Creating new vector store at: {config.PERSIST_DIRECTORY}")
        return Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=config.PERSIST_DIRECTORY
        )
    else:
        config.logger.info(f"Adding {len(documents)} documents to existing vector store")
        existing_store.add_documents(documents)
        return existing_store


def get_document_count(vector_store=None):
    """
    Get the number of documents in the vector store.
    
    Args:
        vector_store: Vector store to check
        
    Returns:
        Number of documents or 0 if vector store doesn't exist
    """
    if vector_store is None:
        vector_store = get_vector_store()
    
    if vector_store is None:
        return 0
    
    try:
        return vector_store._collection.count()
    except Exception as e:
        config.logger.error(f"Error getting document count: {e}")
        return 0


def clear_vector_store():
    """
    Clear the vector store.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if config.os.path.exists(config.PERSIST_DIRECTORY):
            config.logger.info(f"Clearing vector store: {config.PERSIST_DIRECTORY}")
            config.shutil.rmtree(config.PERSIST_DIRECTORY)
            config.os.makedirs(config.PERSIST_DIRECTORY)
            return True
        return False
    except Exception as e:
        config.logger.error(f"Error clearing vector store: {e}")
        return False
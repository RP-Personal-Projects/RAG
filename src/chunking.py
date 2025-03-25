"""
Module for splitting documents into chunks.
"""

from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from . import config


def get_text_splitter(chunk_size: int = None, chunk_overlap: int = None):
    """
    Get a text splitter for chunking documents.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Text splitter instance
    """
    if chunk_size is None:
        chunk_size = config.DEFAULT_CHUNK_SIZE
    
    if chunk_overlap is None:
        chunk_overlap = config.DEFAULT_CHUNK_OVERLAP
    
    config.logger.debug(f"Creating text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )


def split_documents(documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents
    """
    if not documents:
        config.logger.warning("No documents to split")
        return []
    
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    
    config.logger.info(f"Splitting {len(documents)} documents into chunks")
    chunks = text_splitter.split_documents(documents)
    config.logger.info(f"Created {len(chunks)} chunks")
    
    return chunks


def get_splitter_params(text_splitter: RecursiveCharacterTextSplitter) -> Dict[str, Any]:
    """
    Get the parameters of a text splitter.
    
    Args:
        text_splitter: The text splitter to get parameters from
        
    Returns:
        Dictionary of parameters
    """
    return {
        "chunk_size": getattr(text_splitter, '_chunk_size', config.DEFAULT_CHUNK_SIZE),
        "chunk_overlap": getattr(text_splitter, '_chunk_overlap', config.DEFAULT_CHUNK_OVERLAP)
    }
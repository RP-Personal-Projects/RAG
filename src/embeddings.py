"""
Module for handling text embeddings.
"""

from typing import List

try:
    # Try the new package first
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fall back to community package
    from langchain_community.embeddings import HuggingFaceEmbeddings

from . import config


def get_embedding_model(model_name: str = None):
    """
    Get the embedding model.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        The embedding model instance
    """
    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME
        
    try:
        config.logger.info(f"Loading embedding model: {model_name}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name
        )
        return embedding_model
    except Exception as e:
        config.logger.error(f"Error loading embedding model: {e}")
        raise
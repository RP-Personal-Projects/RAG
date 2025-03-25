"""
Module for loading documents from various sources.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, BinaryIO

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader
)

from . import config


def get_loader_for_file(file_path: str):
    """
    Get the appropriate document loader based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Document loader instance
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    config.logger.debug(f"Selecting loader for file type: {file_extension}")
    
    try:
        if file_extension == ".pdf":
            return PyPDFLoader(str(file_path))
        elif file_extension == ".md":
            return UnstructuredMarkdownLoader(str(file_path))
        elif file_extension == ".csv":
            return CSVLoader(str(file_path))
        elif file_extension in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(str(file_path))
        elif file_extension in [".xls", ".xlsx"]:
            return UnstructuredExcelLoader(str(file_path))
        elif file_extension in [".txt", ".text", ".json", ".log"]:
            return TextLoader(str(file_path), autodetect_encoding=True)
        else:
            # Default to text loader for unknown types
            config.logger.warning(f"Unknown file type: {file_extension}, defaulting to TextLoader")
            return TextLoader(str(file_path), autodetect_encoding=True)
    except Exception as e:
        config.logger.error(f"Error creating loader for {file_path}: {e}")
        return None


def load_document(file_path: str) -> List:
    """
    Load a document from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of document objects
    """
    config.logger.info(f"Loading document: {file_path}")
    
    loader = get_loader_for_file(file_path)
    if not loader:
        config.logger.error(f"Failed to create loader for {file_path}")
        return []
    
    try:
        documents = loader.load()
        config.logger.info(f"Loaded {len(documents)} sections from {file_path}")
        return documents
    except Exception as e:
        config.logger.error(f"Error loading {file_path}: {e}")
        return []


def process_uploaded_file(uploaded_file: BinaryIO) -> List:
    """
    Process an uploaded file and return documents.
    
    Args:
        uploaded_file: File object from Streamlit uploader
        
    Returns:
        List of document objects
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        # Write the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load the document
        documents = load_document(tmp_path)
        return documents
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
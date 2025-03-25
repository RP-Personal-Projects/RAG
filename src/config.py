"""
Configuration module for the RAG system.
"""

import os
import logging
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Default LLM
DEFAULT_LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.2

# Retrieval parameters
TOP_K_RESULTS = 4

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_system")
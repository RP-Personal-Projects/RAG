# RAG-Experiments

A flexible framework for experimenting with different Retrieval Augmented Generation (RAG) architectures and techniques, with visualization tools to better understand how RAG works.

## What is RAG?

Retrieval Augmented Generation (RAG) is an AI architecture that enhances Large Language Models (LLMs) by incorporating a retrieval component that fetches relevant information from external knowledge sources. This approach helps ground LLM responses in specific documents and reduces hallucinations by providing factual context.

## Project Structure

```
rag-experiments/
├── src/                 # RAG system implementation
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration parameters
│   ├── chunking.py      # Text chunking implementation
│   ├── document_loader.py # Document loading functionality
│   ├── embeddings.py    # Embedding model implementation
│   ├── vector_store.py  # Vector database operations
│   ├── llm.py           # LLM integration
│   ├── rag_pipeline.py  # RAG pipeline implementation
│   └── webui.py         # Streamlit web interface
├── vis/                 # Visualization tools
│   ├── __init__.py      # Package initialization
│   ├── chunking_visualisation.py  # Chunking visualization tools
│   ├── embedding_visualisation.py # Embedding visualization tools
│   └── vis_app.py       # Visualization web interface
├── data/                # Directory for storing data
├── chroma_db/           # Vector database storage
├── main.py              # Command-line entry point
├── simplified_vis_app.py # Standalone visualization app
├── requirements.txt     # Core dependencies
└── requirements-visualization.txt # Visualization dependencies
```

## System Components

### 1. Core RAG System (`src/`)

* **Chunking**: Splits documents into manageable pieces
* **Embedding**: Converts text to vector representations
* **Vector Store**: Indexes and retrieves embeddings efficiently
* **LLM Integration**: Connects to Ollama for local LLM inference
* **RAG Pipeline**: Combines all components for end-to-end operation
* **Web UI**: User-friendly interface for document management and queries

### 2. Visualization Tools (`vis/`)

* **Chunking Visualization**: Shows how documents are split into chunks
* **Embedding Visualization**: Displays vector representations and similarity
* **Interactive Dashboard**: Helps understand the RAG pipeline's inner workings

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-experiments
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

3. For visualization tools, install additional dependencies:
```bash
pip install -r requirements-visualization.txt
```

4. Install Ollama:
```bash
# For macOS or Linux
curl -fsSL https://ollama.com/install.sh | sh

# For Windows, download from https://ollama.com/download/windows
```

5. Pull a model in Ollama:
```bash
ollama pull llama3
```

## Usage

### Core RAG System

#### Web Interface

To run the Streamlit web interface:

```bash
python main.py web
```

This interface allows you to:
- Upload and process documents
- Select different LLM models
- Ask questions and view answers
- Adjust chunking parameters
- Clear the vector database

#### Command-Line Interface

The system can also be used from the command line:

1. Ingest documents:
```bash
python main.py ingest path/to/documents
```

2. Query the system:
```bash
python main.py query "What is the main topic of these documents?"
```

3. Clear the vector database:
```bash
python main.py clear
```

### Visualization Tools

#### Full Visualization Dashboard

```bash
cd vis
streamlit run vis_app.py
```

This dashboard includes:
- Text chunking visualization: See how documents are split into chunks
- Embedding visualization: View vector representations in 2D/3D space
- Query similarity visualization: Understand how chunks are matched to queries

#### Simplified Embedding Visualization

For a simpler, standalone visualization focused on embeddings:

```bash
streamlit run simplified_vis_app.py
```

This tool helps you understand:
- The chunking process with clear visualization
- The actual numerical values of embeddings
- How embeddings are distributed in vector space
- The similarity between different chunks

## Customization

You can customize various aspects of the system:

1. **Change the embedding model**: Edit `config.py` to use a different model
2. **Adjust chunking parameters**: Update `DEFAULT_CHUNK_SIZE` and `DEFAULT_CHUNK_OVERLAP` in `config.py`
3. **Use different LLM models**: Pull additional models with Ollama and select them in the interface
4. **Visualization techniques**: Choose between PCA, t-SNE, and UMAP for dimensionality reduction

## Requirements

- Python 3.8+
- Ollama installed and running
- At least 8GB RAM recommended (16GB for larger models)
- Internet connection for initial model downloads
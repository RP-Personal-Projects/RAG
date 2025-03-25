"""
Streamlit web interface for the RAG system.

Usage:
    streamlit run src/webui.py
"""

import streamlit as st
import pandas as pd
import time
import shutil

from src import config
from src import llm
from src import vector_store
from src.rag_pipeline import RAGPipeline


def setup_page():
    """Set up the Streamlit page."""
    st.set_page_config(
        page_title="RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 RAG System with Ollama")
    st.markdown("Upload documents, select a model, and ask questions about your data.")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_pipeline' not in st.session_state:
        # Get default model
        available_models = llm.get_available_models()
        default_model = available_models[0] if available_models else config.DEFAULT_LLM_MODEL
        
        # Initialize the RAG pipeline
        try:
            st.session_state.rag_pipeline = RAGPipeline(llm_model=default_model)
            st.sidebar.success(f"✅ System initialized with model: {default_model}")
        except ImportError as e:
            st.sidebar.error(f"❌ Error: {e}")
            st.sidebar.info("Please install langchain-ollama with: pip install langchain-ollama")
            st.stop()
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing system: {e}")
            st.stop()


def setup_sidebar():
    """Set up the sidebar with settings."""
    st.sidebar.title("Settings")
    
    # Model selection
    available_models = llm.get_available_models()
    
    # Get current model
    current_model = st.session_state.rag_pipeline.llm_model
    
    selected_model = st.sidebar.selectbox(
        "Select Ollama Model", 
        available_models,
        index=available_models.index(current_model) if current_model in available_models else 0
    )
    
    # Change model if needed
    if selected_model != current_model:
        with st.sidebar.spinner(f"Loading model {selected_model}..."):
            if st.session_state.rag_pipeline.change_llm(selected_model):
                st.sidebar.success(f"✅ Model changed to {selected_model}")
            else:
                st.sidebar.error(f"❌ Failed to load model {selected_model}")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        # Clear database button
        if st.button("Clear Vector Database", type="primary"):
            if vector_store.clear_vector_store():
                st.sidebar.success("Vector database cleared successfully!")
                # Force a page refresh to reinitialize the system
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to clear vector database")
        
        # Chunk size and overlap
        # Get current values
        chunk_params = st.session_state.rag_pipeline.get_chunk_params()
        current_chunk_size = chunk_params["chunk_size"]
        current_chunk_overlap = chunk_params["chunk_overlap"]
        
        new_chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=4000, 
            value=int(current_chunk_size),
            step=100
        )
        
        new_chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=1000, 
            value=int(current_chunk_overlap),
            step=50
        )
        
        # Update chunking parameters if changed
        if (new_chunk_size != current_chunk_size or new_chunk_overlap != current_chunk_overlap):
            if st.session_state.rag_pipeline.update_chunk_params(new_chunk_size, new_chunk_overlap):
                st.sidebar.success("Chunking parameters updated!")
            else:
                st.sidebar.error("Failed to update chunking parameters")


def upload_tab():
    """Content for the upload tab."""
    st.header("Upload Documents")
    st.markdown("Drag and drop files to add them to the knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Upload documents", 
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "md", "docx", "xlsx"]
    )
    
    if st.button("Process Files", type="primary", disabled=not uploaded_files):
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            # Process each file
            progress_bar = st.progress(0)
            
            # Display file names
            for i, uploaded_file in enumerate(uploaded_files):
                st.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Process all files
            if st.session_state.rag_pipeline.process_uploaded_files(uploaded_files):
                st.success(f"✅ Successfully processed {len(uploaded_files)} files")
                doc_count = st.session_state.rag_pipeline.get_document_count()
                st.info(f"Vector store now contains {doc_count} chunks")
            else:
                st.error("❌ Failed to process files")


def query_tab():
    """Content for the query tab."""
    st.header("Ask Questions")
    
    # Check if we have an initialized system with documents
    doc_count = st.session_state.rag_pipeline.get_document_count()
    if doc_count == 0:
        st.warning("No documents have been ingested yet. Please upload documents first.")
        return
    
    # Input for question
    question = st.text_input("Enter your question")
    
    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Generating answer..."):
            result = st.session_state.rag_pipeline.query(question)
        
        # Display answer
        st.markdown("### Answer")
        st.markdown(result["answer"])
        st.caption(f"Query processed in {result['query_time']:.2f} seconds")
        
        # Display sources
        with st.expander("View Sources"):
            for i, source in enumerate(result["sources"], 1):
                st.markdown(f"**Source {i}**")
                st.text(source[:500] + "..." if len(source) > 500 else source)
                st.divider()


def info_tab():
    """Content for the database info tab."""
    st.header("Database Information")
    
    doc_count = st.session_state.rag_pipeline.get_document_count()
    if doc_count > 0:
        st.metric("Total Chunks", doc_count)
        
        st.success(f"Vector database is active and contains {doc_count} chunks.")
        st.info(f"Database location: {config.PERSIST_DIRECTORY}")
        
        # Display information about chunking
        st.subheader("Chunking Parameters")
        
        # Get current chunk parameters
        chunk_params = st.session_state.rag_pipeline.get_chunk_params()
        
        # Convert all values to strings to avoid type conversion issues with Arrow
        chunk_data = pd.DataFrame({
            "Parameter": ["Chunk Size", "Chunk Overlap", "Embedding Model", "LLM Model"],
            "Value": [
                str(chunk_params["chunk_size"]),
                str(chunk_params["chunk_overlap"]),
                config.EMBEDDING_MODEL_NAME,
                st.session_state.rag_pipeline.llm_model
            ]
        })
        
        st.table(chunk_data)
    else:
        st.warning("No vector database has been created yet.")


def main():
    """Main entry point for the Streamlit app."""
    # Set up the page
    setup_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Set up the sidebar
    setup_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "Ask Questions", "Database Info"])
    
    # Tab content
    with tab1:
        upload_tab()
    
    with tab2:
        query_tab()
    
    with tab3:
        info_tab()


if __name__ == "__main__":
    main()
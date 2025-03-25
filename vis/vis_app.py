"""
Streamlit app for visualizing various aspects of the RAG system.

Usage:
    streamlit run vis_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple

# Add the parent directory to Python path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import visualization functions
from chunking_visualisation import visualize_text_chunks, highlight_chunk_overlaps
from embedding_visualisation import (
    reduce_dimensions_pca, reduce_dimensions_tsne, reduce_dimensions_umap,
    plot_embeddings_2d, plot_embeddings_3d, plot_embedding_heatmap,
    compare_query_to_chunks
)

# Import from the RAG system modules
try:
    from src import config
    from src.chunking import split_documents, get_text_splitter
    from src.document_loader import process_uploaded_file
    from src.embeddings import get_embedding_model
    from src.vector_store import get_vector_store, create_or_update_vector_store
except ImportError:
    st.error("""
    Could not import modules from the src package. Make sure:
    1. The src directory is in the same parent directory as the vis directory
    2. You have all required packages installed
    """)


def setup_page():
    """Set up the Streamlit page."""
    st.set_page_config(
        page_title="RAG Visualization",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 RAG System Visualization")
    st.markdown("Visualize different components of the RAG system to better understand how it works.")


def chunking_tab():
    """Content for the chunking visualization tab."""
    st.header("Text Chunking Visualization")
    st.markdown("""
    This visualization shows how a document is split into chunks for processing by the RAG system.
    Upload a document or use the sample text to see how different chunking parameters affect the result.
    """)
    
    # Input options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample text", value=True)
        
        if use_sample:
            sample_text = st.text_area(
                "Sample text",
                value="""This is a sample document that will be split into chunks. Retrieval Augmented Generation (RAG) systems process documents by breaking them into smaller pieces called chunks. These chunks are then converted into numerical representations called embeddings. When a user asks a question, the system finds the most relevant chunks and uses them to generate an answer. The size of chunks and how much they overlap can significantly impact the quality of results. This visualization helps you understand how the chunking process works in practice.""",
                height=200
            )
            text_to_process = sample_text
        else:
            uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "md", "csv"])
            text_to_process = None
            
            if uploaded_file:
                try:
                    with st.spinner("Processing uploaded file..."):
                        docs = process_uploaded_file(uploaded_file)
                        if docs:
                            text_to_process = docs[0].page_content
                            st.success("File processed successfully!")
                        else:
                            st.error("Failed to process file. Please try a different file.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with col2:
        chunk_size = st.slider("Chunk Size", min_value=50, max_value=2000, value=200, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=50, step=10)
    
    # Process and visualize
    if text_to_process:
        try:
            # Create text splitter
            text_splitter = get_text_splitter(chunk_size, chunk_overlap)
            
            # Split text into chunks
            chunks = text_splitter.split_text(text_to_process)
            
            # Show stats
            st.markdown(f"**Document Statistics:**")
            st.markdown(f"- Document length: {len(text_to_process)} characters")
            st.markdown(f"- Number of chunks: {len(chunks)}")
            st.markdown(f"- Average chunk length: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} characters")
            
            # Show chunks
            with st.expander("View Chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:** ({len(chunk)} chars)")
                    st.text(chunk)
            
            # Visualize chunks
            st.subheader("Chunk Visualization")
            
            try:
                chunk_fig = visualize_text_chunks(text_to_process, chunks)
                if chunk_fig is not None:
                    st.plotly_chart(chunk_fig, use_container_width=True)
                else:
                    st.warning("Could not generate chunk visualization")
            except Exception as e:
                st.error(f"Error generating chunk visualization: {str(e)}")
            
            # Visualize overlaps
            if len(chunks) >= 2:  # Only show overlap visualization if there are at least 2 chunks
                st.subheader("Chunk Overlap Visualization")
                try:
                    overlap_fig = highlight_chunk_overlaps(chunks, chunk_size, chunk_overlap)
                    if overlap_fig is not None:
                        st.plotly_chart(overlap_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate overlap visualization")
                except Exception as e:
                    st.error(f"Error generating overlap visualization: {str(e)}")
            else:
                st.info("Need at least 2 chunks to visualize overlaps")
                
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")


def embedding_tab():
    """Content for the embedding visualization tab."""
    st.header("Embedding Visualization")
    st.markdown("""
    This visualization shows how document chunks are represented as vectors in embedding space.
    Upload a document or use the sample text to see how chunks are embedded and how similarity is calculated.
    """)
    
    # Input options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample text", value=True, key="embed_sample")
        
        if use_sample:
            sample_text = st.text_area(
                "Sample text",
                value="""This is a sample document about machine learning. Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers. Natural language processing is another field of AI that focuses on the interaction between computers and human language. Computer vision is the field of AI that enables computers to derive information from digital images and videos. Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.""",
                height=200,
                key="embed_text_area"
            )
            text_to_process = sample_text
        else:
            uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "md", "csv"], key="embed_uploader")
            text_to_process = None
            
            if uploaded_file:
                try:
                    with st.spinner("Processing uploaded file..."):
                        docs = process_uploaded_file(uploaded_file)
                        if docs:
                            text_to_process = docs[0].page_content
                            st.success("File processed successfully!")
                        else:
                            st.error("Failed to process file. Please try a different file.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with col2:
        chunk_size = st.slider("Chunk Size", min_value=50, max_value=2000, value=200, step=50, key="embed_chunk_size")
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=50, step=10, key="embed_chunk_overlap")
        
        dim_reduction = st.selectbox(
            "Dimensionality Reduction",
            options=["PCA", "t-SNE", "UMAP"],
            index=0
        )
        
        viz_dimensions = st.radio("Visualization Dimensions", options=["2D", "3D"], index=0)
    
    if text_to_process:
        try:
            # Create text splitter
            text_splitter = get_text_splitter(chunk_size, chunk_overlap)
            
            # Split text into chunks
            chunks = text_splitter.split_text(text_to_process)
            
            if not chunks:
                st.warning("No chunks were created. Try different chunking parameters.")
                return
            
            # Load embedding model
            with st.spinner("Loading embedding model..."):
                try:
                    embedding_model = get_embedding_model()
                except Exception as e:
                    st.error(f"Error loading embedding model: {str(e)}")
                    return
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                try:
                    embeddings = embedding_model.embed_documents(chunks)
                    embeddings_array = np.array(embeddings)
                    
                    st.markdown(f"**Embedding Dimensions:** {embeddings_array.shape}")
                    
                    # Let user enter a query
                    query = st.text_input("Enter a query to see similarity:", "What is machine learning?")
                    
                    if query:
                        query_embedding = embedding_model.embed_query(query)
                        query_embedding_array = np.array(query_embedding)
                        
                        # Calculate similarity scores
                        similarities = []
                        for embedding in embeddings:
                            # Cosine similarity
                            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                            similarities.append(float(similarity))
                        
                        # Get top chunks
                        top_k = min(3, len(chunks))
                        top_indices = np.argsort(similarities)[-top_k:][::-1]
                        
                        st.markdown("### Top Similar Chunks")
                        for i, idx in enumerate(top_indices):
                            st.markdown(f"**Chunk {idx+1}** (Similarity: {similarities[idx]:.4f})")
                            st.text(chunks[idx])
                except Exception as e:
                    st.error(f"Error generating embeddings: {str(e)}")
                    return
            
            # Reduce dimensions for visualization
            with st.spinner(f"Reducing dimensions using {dim_reduction}..."):
                try:
                    n_dims = 3 if viz_dimensions == "3D" else 2
                    
                    if dim_reduction == "t-SNE":
                        reduced_embeddings = reduce_dimensions_tsne(embeddings_array, n_components=n_dims)
                    elif dim_reduction == "UMAP":
                        reduced_embeddings = reduce_dimensions_umap(embeddings_array, n_components=n_dims)
                    else:  # PCA
                        reduced_embeddings = reduce_dimensions_pca(embeddings_array, n_components=n_dims)
                except Exception as e:
                    st.error(f"Error reducing dimensions: {str(e)}")
                    return
            
            # Visualize embeddings
            st.subheader(f"{viz_dimensions} Embedding Visualization")
            
            try:
                if query:
                    # Reduce dimensions for the query too
                    if dim_reduction == "t-SNE":
                        # For t-SNE, we need to combine and reduce together to maintain relationships
                        combined = np.vstack([query_embedding_array, embeddings_array])
                        combined_reduced = reduce_dimensions_tsne(combined, n_components=n_dims)
                        query_reduced = combined_reduced[0].reshape(1, -1)
                        chunks_reduced = combined_reduced[1:]
                    elif dim_reduction == "UMAP":
                        # Same for UMAP
                        combined = np.vstack([query_embedding_array, embeddings_array])
                        combined_reduced = reduce_dimensions_umap(combined, n_components=n_dims)
                        query_reduced = combined_reduced[0].reshape(1, -1)
                        chunks_reduced = combined_reduced[1:]
                    else:  # PCA
                        # PCA can transform new points
                        pca = PCA(n_components=n_dims)
                        pca.fit(embeddings_array)
                        query_reduced = pca.transform(query_embedding_array.reshape(1, -1))
                        chunks_reduced = pca.transform(embeddings_array)
                    
                    # Create comparison visualization
                    embedding_fig, similarity_fig = compare_query_to_chunks(
                        query_reduced[0], 
                        chunks_reduced,
                        chunks,
                        similarities,
                        top_indices
                    )
                    
                    if embedding_fig is not None:
                        st.plotly_chart(embedding_fig, use_container_width=True)
                    
                    if similarity_fig is not None:
                        st.subheader("Similarity Scores")
                        st.plotly_chart(similarity_fig, use_container_width=True)
                else:
                    # Just visualize the chunks
                    chunk_labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                    
                    if viz_dimensions == "3D":
                        fig = plot_embeddings_3d(reduced_embeddings, labels=chunk_labels)
                    else:
                        fig = plot_embeddings_2d(reduced_embeddings, labels=chunk_labels)
                    
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate embedding visualization")
                
                # Show embedding heatmap
                with st.expander("View Embedding Values Heatmap"):
                    heatmap_fig = plot_embedding_heatmap(embeddings_array)
                    if heatmap_fig is not None:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate heatmap visualization")
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        except Exception as e:
            st.error(f"Error in embedding tab: {str(e)}")


def main():
    """Main entry point for the visualization app."""
    # Set up the page
    setup_page()
    
    # Initialize session state if needed
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Chunking"
    
    # Create tabs
    tab1, tab2 = st.tabs(["Text Chunking", "Embedding Visualization"])
    
    # Tab content
    with tab1:
        chunking_tab()
    
    with tab2:
        embedding_tab()


if __name__ == "__main__":
    main()
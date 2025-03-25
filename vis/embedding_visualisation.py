"""
Module for visualizing embeddings from the RAG system.

This module provides functions to visualize high-dimensional embeddings
in lower-dimensional space using techniques like PCA, t-SNE, and UMAP.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import umap

from typing import List, Dict, Any, Optional, Union, Tuple


def reduce_dimensions_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using PCA.
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to
        
    Returns:
        Reduced embeddings with shape (n_samples, n_components)
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)


def reduce_dimensions_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using t-SNE.
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to
        perplexity: t-SNE perplexity parameter (related to number of nearest neighbors)
        
    Returns:
        Reduced embeddings with shape (n_samples, n_components)
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    return tsne.fit_transform(embeddings)


def reduce_dimensions_umap(embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using UMAP.
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to
        n_neighbors: UMAP n_neighbors parameter (size of local neighborhood)
        
    Returns:
        Reduced embeddings with shape (n_samples, n_components)
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_embeddings_2d(
    embeddings_2d: np.ndarray, 
    labels: Optional[List[str]] = None, 
    title: str = "Embedding Visualization",
    color_by: Optional[Union[List[float], List[int]]] = None,
    color_name: str = "Value",
    point_sizes: Optional[List[float]] = None,
    highlight_indices: Optional[List[int]] = None,
    interactive: bool = True
) -> Any:
    """
    Plot 2D embedding projections.
    
    Args:
        embeddings_2d: Array of 2D embeddings with shape (n_samples, 2)
        labels: Optional list of labels for each point
        title: Plot title
        color_by: Optional values to color points by
        color_name: Name of the color dimension for the legend
        point_sizes: Optional list of point sizes
        highlight_indices: Optional list of indices to highlight
        interactive: If True, return a Plotly figure; if False, return a Matplotlib figure
        
    Returns:
        Plotly figure or Matplotlib figure
    """
    if interactive:
        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels if labels is not None else [f"Point {i}" for i in range(len(embeddings_2d))]
        })
        
        if color_by is not None:
            df['color'] = color_by
            fig = px.scatter(
                df, x='x', y='y', 
                color='color', 
                hover_name='label',
                color_continuous_scale=px.colors.sequential.Viridis,
                title=title
            )
            fig.update_layout(coloraxis_colorbar=dict(title=color_name))
        else:
            fig = px.scatter(
                df, x='x', y='y', 
                hover_name='label',
                title=title
            )
        
        # Set point sizes if provided
        if point_sizes is not None:
            fig.update_traces(marker=dict(size=point_sizes))
        
        # Highlight specific points if requested
        if highlight_indices is not None:
            highlight_x = [embeddings_2d[i, 0] for i in highlight_indices]
            highlight_y = [embeddings_2d[i, 1] for i in highlight_indices]
            highlight_labels = [labels[i] if labels is not None else f"Point {i}" for i in highlight_indices]
            
            fig.add_trace(go.Scatter(
                x=highlight_x, 
                y=highlight_y,
                mode='markers',
                marker=dict(
                    color='red',
                    size=15,
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                name='Highlighted',
                text=highlight_labels,
                hoverinfo='text'
            ))
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Legend",
            font=dict(
                family="Arial, sans-serif",
                size=14
            )
        )
        
        return fig
    else:
        # Use Matplotlib for static plotting
        plt.figure(figsize=(10, 8))
        
        # Basic scatter plot
        if color_by is not None:
            scatter = plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=color_by, 
                cmap='viridis', 
                alpha=0.7,
                s=point_sizes if point_sizes is not None else 50
            )
            plt.colorbar(scatter, label=color_name)
        else:
            plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                alpha=0.7,
                s=point_sizes if point_sizes is not None else 50
            )
        
        # Highlight specific points if requested
        if highlight_indices is not None:
            highlight_x = [embeddings_2d[i, 0] for i in highlight_indices]
            highlight_y = [embeddings_2d[i, 1] for i in highlight_indices]
            plt.scatter(
                highlight_x, 
                highlight_y, 
                color='red', 
                s=150, 
                edgecolors='black', 
                linewidth=2,
                alpha=0.8,
                label='Highlighted'
            )
        
        # Add labels if provided
        if labels is not None and len(labels) <= 30:  # Only show labels if not too many points
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        
        return plt.gcf()


def plot_embeddings_3d(
    embeddings_3d: np.ndarray, 
    labels: Optional[List[str]] = None, 
    title: str = "3D Embedding Visualization",
    color_by: Optional[Union[List[float], List[int]]] = None,
    color_name: str = "Value",
    point_sizes: Optional[List[float]] = None,
    highlight_indices: Optional[List[int]] = None
) -> Any:
    """
    Plot 3D embedding projections using Plotly.
    
    Args:
        embeddings_3d: Array of 3D embeddings with shape (n_samples, 3)
        labels: Optional list of labels for each point
        title: Plot title
        color_by: Optional values to color points by
        color_name: Name of the color dimension for the legend
        point_sizes: Optional list of point sizes
        highlight_indices: Optional list of indices to highlight
        
    Returns:
        Plotly figure
    """
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels if labels is not None else [f"Point {i}" for i in range(len(embeddings_3d))]
    })
    
    if color_by is not None:
        df['color'] = color_by
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='color',
            hover_name='label',
            color_continuous_scale=px.colors.sequential.Viridis,
            title=title
        )
        fig.update_layout(coloraxis_colorbar=dict(title=color_name))
    else:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            hover_name='label',
            title=title
        )
    
    # Set point sizes if provided
    if point_sizes is not None:
        fig.update_traces(marker=dict(size=point_sizes))
    
    # Highlight specific points if requested
    if highlight_indices is not None:
        highlight_x = [embeddings_3d[i, 0] for i in highlight_indices]
        highlight_y = [embeddings_3d[i, 1] for i in highlight_indices]
        highlight_z = [embeddings_3d[i, 2] for i in highlight_indices]
        highlight_labels = [labels[i] if labels is not None else f"Point {i}" for i in highlight_indices]
        
        fig.add_trace(go.Scatter3d(
            x=highlight_x, 
            y=highlight_y,
            z=highlight_z,
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                line=dict(
                    color='black',
                    width=2
                )
            ),
            name='Highlighted',
            text=highlight_labels,
            hoverinfo='text'
        ))
    
    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        legend_title="Legend",
        font=dict(
            family="Arial, sans-serif",
            size=14
        )
    )
    
    return fig


def plot_embedding_heatmap(
    embeddings: np.ndarray, 
    labels: Optional[List[str]] = None,
    title: str = "Embedding Values Heatmap",
    max_features: int = 50
) -> Any:
    """
    Create a heatmap visualization of embedding values.
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        labels: Optional list of labels for each sample
        title: Plot title
        max_features: Maximum number of features to display
        
    Returns:
        Plotly figure
    """
    # If there are too many features, sample them
    if embeddings.shape[1] > max_features:
        # Randomly select features to visualize
        selected_features = np.random.choice(embeddings.shape[1], max_features, replace=False)
        embeddings_subset = embeddings[:, selected_features]
        feature_labels = [f"Dim {i}" for i in selected_features]
    else:
        embeddings_subset = embeddings
        feature_labels = [f"Dim {i}" for i in range(embeddings.shape[1])]
    
    # If there are too many samples, sample them
    max_samples = 50
    if embeddings.shape[0] > max_samples:
        # Randomly select samples to visualize
        selected_samples = np.random.choice(embeddings.shape[0], max_samples, replace=False)
        embeddings_subset = embeddings_subset[selected_samples, :]
        sample_labels = [labels[i] if labels is not None else f"Sample {i}" for i in selected_samples]
    else:
        sample_labels = labels if labels is not None else [f"Sample {i}" for i in range(embeddings.shape[0])]
    
    # Create heatmap
    fig = px.imshow(
        embeddings_subset,
        labels=dict(x="Embedding Dimension", y="Sample", color="Value"),
        x=feature_labels,
        y=sample_labels,
        color_continuous_scale="Viridis",
        title=title
    )
    
    # Add colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title="Value"))
    
    return fig


def compare_query_to_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_texts: List[str],
    similarity_scores: Optional[List[float]] = None,
    highlight_indices: Optional[List[int]] = None,
    reduction_method: str = "pca"
) -> Tuple[Any, Any]:
    """
    Visualize the similarity between a query and document chunks in embedding space.
    
    Args:
        query_embedding: Query embedding vector with shape (n_features,)
        chunk_embeddings: Chunk embeddings with shape (n_chunks, n_features)
        chunk_texts: Text of each chunk
        similarity_scores: Optional similarity scores between query and each chunk
        highlight_indices: Optional indices of chunks to highlight
        reduction_method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        
    Returns:
        Tuple of (embedding_plot, similarity_plot)
    """
    # Combine query and chunk embeddings
    all_embeddings = np.vstack([query_embedding.reshape(1, -1), chunk_embeddings])
    
    # Create labels
    labels = ["Query"] + [f"Chunk {i}" for i in range(len(chunk_embeddings))]
    
    # Set colors to differentiate query from chunks
    colors = ["query" if i == 0 else "chunk" for i in range(len(all_embeddings))]
    
    # Reduce dimensions
    if reduction_method == "tsne":
        reduced_embeddings = reduce_dimensions_tsne(all_embeddings)
    elif reduction_method == "umap":
        reduced_embeddings = reduce_dimensions_umap(all_embeddings)
    else:  # default to PCA
        reduced_embeddings = reduce_dimensions_pca(all_embeddings)
    
    # Create the embeddings visualization
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'type': colors
    })
    
    # Set sizes: larger for query, conditionally sized for chunks based on similarity
    sizes = [15]  # Query size
    if similarity_scores is not None:
        # Scale similarity scores to point sizes (between 5 and 12)
        min_size, max_size = 5, 12
        sizes.extend([min_size + (max_size - min_size) * score for score in similarity_scores])
    else:
        sizes.extend([7] * len(chunk_embeddings))  # Default chunk size
    
    df['size'] = sizes
    
    # Create plot
    embedding_fig = px.scatter(
        df, x='x', y='y', 
        color='type',
        symbol='type',
        size='size',
        hover_name='label',
        title="Query vs. Chunks in Embedding Space",
        color_discrete_map={"query": "red", "chunk": "blue"}
    )
    
    embedding_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    # Highlight specific chunks if requested
    if highlight_indices is not None:
        highlight_indices_adjusted = [i+1 for i in highlight_indices]  # Adjust for query at index 0
        highlight_x = [reduced_embeddings[i, 0] for i in highlight_indices_adjusted]
        highlight_y = [reduced_embeddings[i, 1] for i in highlight_indices_adjusted]
        highlight_labels = [labels[i] for i in highlight_indices_adjusted]
        
        embedding_fig.add_trace(go.Scatter(
            x=highlight_x, 
            y=highlight_y,
            mode='markers',
            marker=dict(
                color='yellow',
                size=15,
                line=dict(
                    color='black',
                    width=2
                ),
                symbol='star'
            ),
            name='Retrieved',
            text=highlight_labels,
            hoverinfo='text'
        ))
    
    # Create a bar chart of similarity scores if provided
    if similarity_scores is not None:
        sim_df = pd.DataFrame({
            'Chunk': [f"Chunk {i}" for i in range(len(similarity_scores))],
            'Similarity': similarity_scores
        })
        sim_df = sim_df.sort_values('Similarity', ascending=False)
        
        similarity_fig = px.bar(
            sim_df, x='Chunk', y='Similarity', 
            title="Similarity Scores",
            color='Similarity',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Highlight selected chunks if requested
        if highlight_indices is not None:
            highlight_colors = ['rgba(255, 255, 0, 0.7)' if i in highlight_indices else 'rgba(0, 0, 0, 0.0)' 
                              for i in range(len(similarity_scores))]
            
            sim_df['highlight'] = highlight_colors
            similarity_fig.update_traces(marker_line_color=highlight_colors, marker_line_width=3)
    else:
        similarity_fig = None
    
    return embedding_fig, similarity_fig
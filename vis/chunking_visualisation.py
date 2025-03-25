"""
Module for visualizing text chunking in the RAG system.

This module provides functions to visualize how text documents are
split into chunks for processing in the RAG pipeline.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Tuple


def visualize_text_chunks(
    original_text: str,
    chunks: List[str],
    chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    title: str = "Document Chunking Visualization"
) -> Any:
    """
    Create a visualization showing how the original text is divided into chunks.
    
    Args:
        original_text: The original document text
        chunks: List of chunk texts
        chunk_metadata: Optional list of metadata for each chunk
        title: Title for the visualization
        
    Returns:
        Plotly figure object
    """
    # Create figure with two rows: one for original text, one for chunks
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=["Original Document", "Chunked Document"],
        vertical_spacing=0.1,
        row_heights=[0.3, 0.7]
    )
    
    # Add original text as a single chunk
    fig.add_trace(
        go.Scatter(
            x=[0.5], 
            y=[0.5],
            mode='text',
            text=[original_text[:100] + "..." if len(original_text) > 100 else original_text],
            textposition="middle center",
            textfont=dict(size=10),
            hoverinfo='text',
            hovertext=f"Full document: {len(original_text)} characters",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Prepare chunk visualization data
    chunk_y_positions = []
    chunk_heights = []
    chunk_colors = []
    chunk_texts = []
    hover_texts = []
    
    # Assign colors to chunks in a gradient
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    colormap = cm.get_cmap('viridis', len(chunks))
    color_list = [mcolors.rgb2hex(colormap(i)) for i in range(len(chunks))]
    
    # Calculate positions and colors for each chunk
    for i, chunk in enumerate(chunks):
        chunk_y_positions.append(1.0 - (i+0.5)/len(chunks))
        chunk_heights.append(0.9/len(chunks))
        chunk_colors.append(color_list[i])
        
        # Prepare display text (shortened if needed)
        display_text = chunk[:50] + "..." if len(chunk) > 50 else chunk
        chunk_texts.append(display_text)
        
        # Prepare hover text with metadata if available
        if chunk_metadata and i < len(chunk_metadata):
            meta_str = ", ".join([f"{k}: {v}" for k, v in chunk_metadata[i].items()])
            hover_texts.append(f"Chunk {i+1} ({len(chunk)} chars)<br>{meta_str}<br>{chunk[:200]}...")
        else:
            hover_texts.append(f"Chunk {i+1} ({len(chunk)} chars)<br>{chunk[:200]}...")
    
    # Add rectangles for chunks
    for i in range(len(chunks)):
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 1, 0, 0],  # Rectangle coordinates
                y=[chunk_y_positions[i] - chunk_heights[i]/2, 
                   chunk_y_positions[i] - chunk_heights[i]/2, 
                   chunk_y_positions[i] + chunk_heights[i]/2, 
                   chunk_y_positions[i] + chunk_heights[i]/2, 
                   chunk_y_positions[i] - chunk_heights[i]/2],
                mode='lines',
                fill='toself',
                fillcolor=chunk_colors[i],
                line=dict(color='black', width=1),
                name=f"Chunk {i+1}",
                text=chunk_texts[i],
                hoverinfo='text',
                hovertext=hover_texts[i]
            ),
            row=2, col=1
        )
        
        # Add chunk text
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[chunk_y_positions[i]],
                mode='text',
                text=[chunk_texts[i]],
                textposition="middle center",
                textfont=dict(size=10, color='white' if i > len(chunks)/2 else 'black'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=150 + 100*len(chunks),
        width=800,
        template="plotly_white",
        showlegend=True
    )
    
    # Remove axis labels and ticks
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    
    return fig


def highlight_chunk_overlaps(
    chunks: List[str],
    chunk_size: int,
    chunk_overlap: int,
    display_length: int = 100,
    title: str = "Chunk Overlap Visualization"
) -> Any:
    """
    Create a visualization highlighting the overlapping regions between adjacent chunks.
    
    Args:
        chunks: List of chunk texts
        chunk_size: The target size of each chunk
        chunk_overlap: The target overlap between chunks
        display_length: Maximum length of text to display in the visualization
        title: Title for the visualization
        
    Returns:
        Plotly figure object
    """
    # Prepare data for visualization
    data = []
    
    for i, chunk in enumerate(chunks):
        # Trim chunk for display if needed
        display_chunk = chunk[:display_length] + "..." if len(chunk) > display_length else chunk
        
        # Try to identify overlap with next chunk
        overlap_text = ""
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            
            # Find the suffix of current chunk that appears as prefix in next chunk
            # Start with maximum possible overlap and decrease
            actual_overlap = min(len(chunk), len(next_chunk), chunk_overlap)
            while actual_overlap > 0:
                suffix = chunk[-actual_overlap:]
                prefix = next_chunk[:actual_overlap]
                if suffix == prefix:
                    overlap_text = suffix
                    break
                actual_overlap -= 1
        
        # Create data entry
        entry = {
            "index": i,
            "chunk": display_chunk,
            "length": len(chunk),
            "target_size": chunk_size,
            "diff_from_target": len(chunk) - chunk_size,
            "overlap_text": overlap_text,
            "overlap_size": len(overlap_text) if overlap_text else 0
        }
        data.append(entry)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate positions
    y_step = 1.0 / (len(chunks) + 1)
    y_positions = [1.0 - (i + 1) * y_step for i in range(len(chunks))]
    
    # Add main chunk rectangles
    for i, entry in enumerate(data):
        # Normalize chunk length to display width
        width = min(0.8, 0.3 + 0.5 * entry["length"] / chunk_size)
        
        # Add rectangle for chunk
        fig.add_trace(go.Scatter(
            x=[0, width, width, 0, 0],  # Rectangle coordinates
            y=[y_positions[i] - 0.03, 
               y_positions[i] - 0.03, 
               y_positions[i] + 0.03, 
               y_positions[i] + 0.03, 
               y_positions[i] - 0.03],
            mode='lines',
            fill='toself',
            fillcolor='lightblue',
            line=dict(color='navy', width=1),
            name=f"Chunk {i+1}",
            hoverinfo='text',
            hovertext=f"Chunk {i+1}<br>Length: {entry['length']} chars<br>Diff from target: {entry['diff_from_target']} chars"
        ))
        
        # Add text label
        fig.add_trace(go.Scatter(
            x=[width/2],
            y=[y_positions[i]],
            mode='text',
            text=[f"Chunk {i+1}"],
            textposition="middle center",
            textfont=dict(size=10),
            showlegend=False
        ))
    
    # Add overlap indicators
    for i, entry in enumerate(data):
        if i < len(data) - 1 and entry["overlap_size"] > 0:
            # Calculate position for overlap indicator
            width = min(0.8, 0.3 + 0.5 * entry["length"] / chunk_size)
            overlap_width = 0.1 * entry["overlap_size"] / chunk_overlap
            
            # Add rectangle for overlap
            fig.add_trace(go.Scatter(
                x=[width - overlap_width, width, width, width - overlap_width, width - overlap_width],
                y=[y_positions[i] - 0.03, 
                   y_positions[i] - 0.03, 
                   y_positions[i] + 0.03, 
                   y_positions[i] + 0.03, 
                   y_positions[i] - 0.03],
                mode='lines',
                fill='toself',
                fillcolor='yellow',
                line=dict(color='orange', width=1),
                name=f"Overlap {i+1}-{i+2}",
                hoverinfo='text',
                hovertext=f"Overlap between Chunk {i+1} and {i+2}<br>Size: {entry['overlap_size']} chars<br>Text: {entry['overlap_text']}"
            ))
            
            # Add connecting line to next chunk
            fig.add_trace(go.Scatter(
                x=[width - overlap_width/2, 0 + overlap_width/2],
                y=[y_positions[i], y_positions[i+1]],
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                showlegend=False
            ))
            
            # Add overlap indicator to next chunk
            fig.add_trace(go.Scatter(
                x=[0, 0 + overlap_width, 0 + overlap_width, 0, 0],
                y=[y_positions[i+1] - 0.03, 
                   y_positions[i+1] - 0.03, 
                   y_positions[i+1] + 0.03, 
                   y_positions[i+1] + 0.03, 
                   y_positions[i+1] - 0.03],
                mode='lines',
                fill='toself',
                fillcolor='yellow',
                line=dict(color='orange', width=1),
                showlegend=False
            ))
    
    #
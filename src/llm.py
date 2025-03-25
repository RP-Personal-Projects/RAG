"""
Module for LLM integration.
"""

import subprocess
from typing import List, Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
    except ImportError:
        OllamaLLM = None

from . import config


def get_llm(model_name: Optional[str] = None, temperature: Optional[float] = None):
    """
    Get an LLM instance.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        
    Returns:
        LLM instance
    """
    if model_name is None:
        model_name = config.DEFAULT_LLM_MODEL
    
    if temperature is None:
        temperature = config.LLM_TEMPERATURE
    
    if OllamaLLM is None:
        config.logger.error("OllamaLLM not available. Install with: pip install langchain-ollama")
        raise ImportError("langchain-ollama is not installed")
    
    try:
        config.logger.info(f"Initializing Ollama LLM with model: {model_name}")
        return OllamaLLM(model=model_name, temperature=temperature)
    except Exception as e:
        config.logger.error(f"Error initializing Ollama LLM: {e}")
        raise


def get_available_models() -> List[str]:
    """
    Get list of available Ollama models.
    
    Returns:
        List of model names
    """
    try:
        # Run the ollama list command
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse the output to get model names
        lines = result.stdout.strip().split('\n')
        models = []
        
        # Skip the header line if it exists
        start_idx = 1 if len(lines) > 0 and 'NAME' in lines[0] else 0
        
        for line in lines[start_idx:]:
            if line.strip():
                # Extract the first part which is the model name
                parts = line.split()
                if parts:
                    models.append(parts[0])
        
        return models
    except Exception as e:
        config.logger.error(f"Error getting models: {e}")
        # Return some default models
        return ["llama3", "mistral", "gemma"]
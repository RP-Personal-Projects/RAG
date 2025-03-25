"""
Module for the complete RAG pipeline.
"""

import time
from typing import List, Dict, Any, Optional

from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from . import config
from . import embeddings
from . import document_loader
from . import chunking
from . import vector_store
from . import llm


class RAGPipeline:
    """RAG pipeline integrating all components."""
    
    def __init__(self, 
                 llm_model: Optional[str] = None, 
                 embedding_model: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm_model: Model name for the LLM
            embedding_model: Model name for embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = chunking.get_text_splitter(chunk_size, chunk_overlap)
        self.embedding_model = embeddings.get_embedding_model(embedding_model)
        self.vectorstore = vector_store.get_vector_store(self.embedding_model)
        self.llm_model = llm_model or config.DEFAULT_LLM_MODEL
        self.llm_instance = None
        self.qa_chain = None
        
        # Initialize LLM and QA chain
        self._initialize_llm()
        if self.vectorstore is not None:
            self._create_qa_chain()
    
    def _initialize_llm(self):
        """Initialize the LLM."""
        try:
            self.llm_instance = llm.get_llm(self.llm_model)
        except Exception as e:
            config.logger.error(f"Failed to initialize LLM: {e}")
            self.llm_instance = None
    
    def change_llm(self, new_model: str):
        """
        Change the LLM model.
        
        Args:
            new_model: New model name
            
        Returns:
            True if successful, False otherwise
        """
        if new_model == self.llm_model:
            return True
        
        try:
            self.llm_model = new_model
            self.llm_instance = llm.get_llm(new_model)
            
            # Recreate QA chain
            if self.vectorstore is not None:
                self._create_qa_chain()
            
            return True
        except Exception as e:
            config.logger.error(f"Error changing LLM model: {e}")
            return False
    
    def _create_qa_chain(self):
        """Create the QA chain for retrieval and generation."""
        if self.vectorstore is None or self.llm_instance is None:
            config.logger.warning("Cannot create QA chain: missing vectorstore or LLM")
            self.qa_chain = None
            return
        
        # Custom prompt template
        prompt_template = """
        Answer the question based ONLY on the following context. DO NOT use any prior knowledge:
        {context}

        Question: {question}

        When answering:
        - If the exact answer is in the context, provide it word-for-word from the context.
        - If the answer is not in the context, respond ONLY with: "I don't have information about that in the provided documents."
        - Do not elaborate or add any information not explicitly stated in the context.
        - Always cite the exact text from the context that contains your answer.

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K_RESULTS}
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_instance,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        config.logger.info("QA chain created successfully")
    
    def process_documents(self, documents: List[Document]) -> bool:
        """
        Process documents and add them to the vector store.
        
        Args:
            documents: List of documents to process
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            config.logger.warning("No documents to process")
            return False
        
        try:
            # Split documents into chunks
            chunks = chunking.split_documents(
                documents, 
                getattr(self.text_splitter, '_chunk_size', None),
                getattr(self.text_splitter, '_chunk_overlap', None)
            )
            
            if not chunks:
                config.logger.warning("No chunks created from documents")
                return False
            
            # Add to vector store
            self.vectorstore = vector_store.create_or_update_vector_store(
                chunks, 
                self.embedding_model
            )
            
            # Create or update QA chain
            self._create_qa_chain()
            
            return True
        except Exception as e:
            config.logger.error(f"Error processing documents: {e}")
            return False
    
    def process_files(self, file_paths: List[str]) -> bool:
        """
        Process files and add them to the vector store.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            True if successful, False otherwise
        """
        all_docs = []
        
        for file_path in file_paths:
            docs = document_loader.load_document(file_path)
            all_docs.extend(docs)
        
        return self.process_documents(all_docs)
    
    def process_uploaded_files(self, uploaded_files) -> bool:
        """
        Process uploaded files and add them to the vector store.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            True if successful, False otherwise
        """
        all_docs = []
        
        for uploaded_file in uploaded_files:
            docs = document_loader.process_uploaded_file(uploaded_file)
            all_docs.extend(docs)
        
        return self.process_documents(all_docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            config.logger.warning("QA chain not initialized for querying")
            return {
                "answer": "No documents have been ingested yet. Please upload documents first.",
                "sources": [],
                "query_time": 0
            }
        
        try:
            start_time = time.time()
            result = self.qa_chain.invoke({"query": question})
            query_time = time.time() - start_time
            
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]],
                "query_time": query_time
            }
        except Exception as e:
            config.logger.error(f"Error during query: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "query_time": 0
            }
    
    def get_chunk_params(self) -> Dict[str, Any]:
        """
        Get chunking parameters.
        
        Returns:
            Dictionary of chunking parameters
        """
        if self.text_splitter:
            return chunking.get_splitter_params(self.text_splitter)
        return {
            "chunk_size": config.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": config.DEFAULT_CHUNK_OVERLAP
        }
    
    def update_chunk_params(self, chunk_size: int, chunk_overlap: int) -> bool:
        """
        Update chunking parameters.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.text_splitter = chunking.get_text_splitter(chunk_size, chunk_overlap)
            return True
        except Exception as e:
            config.logger.error(f"Error updating chunk parameters: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        return vector_store.get_document_count(self.vectorstore)
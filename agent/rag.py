"""
RAG Pipeline for AEGIS Cognitive Tutor
Handles document loading, chunking, embeddings, and retrieval.
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from typing import Tuple, List


def setup_rag_pipeline(file_path: str) -> Tuple:
    """
    Set up the RAG pipeline for a given document.
    
    Args:
        file_path: Path to the PDF or TXT file
        
    Returns:
        Tuple of (retriever, full_text, segments)
    """
    # Load document based on extension
    extension = file_path.lower().split('.')[-1]
    
    if extension == 'pdf':
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    
    # Combine all text
    full_text = " ".join([doc.page_content for doc in documents])
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create segments for tracking
    segments = [chunk.page_content for chunk in chunks]
    
    # Setup embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create ChromaDB client and collection
    client = chromadb.Client()
    collection = client.create_collection(name="aegis_docs")
    
    # Add documents to collection
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"source": file_path, "chunk_id": i}],
            ids=[f"chunk_{i}"]
        )
    
    # Create retriever
    retriever = collection.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    return retriever, full_text, segments

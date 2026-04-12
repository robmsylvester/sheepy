"""Retrieval-augmented classification module."""
from sheepy.retrieval.retriever import BM25Retriever, DenseRetriever
from sheepy.retrieval.rag_pipeline import RAGPipeline

__all__ = ["DenseRetriever", "BM25Retriever", "RAGPipeline"]

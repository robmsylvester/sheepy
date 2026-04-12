"""Tests for sheepy.retrieval module."""
import numpy as np
import pytest


class TestDenseRetriever:
    def test_index_and_retrieve(self):
        from sheepy.retrieval.retriever import DenseRetriever

        def embed(texts):
            vecs = []
            for t in texts:
                rng = np.random.RandomState(hash(t) % 2**31)
                vecs.append(rng.randn(16).astype(np.float32))
            return np.array(vecs)

        retriever = DenseRetriever(embed_fn=embed)
        docs = ["the cat sat", "the dog ran", "birds fly high"]
        labels = ["cat", "dog", "bird"]
        retriever.index(docs, labels)

        results = retriever.retrieve("the cat sat", k=2)
        assert len(results) == 2
        assert results[0]["text"] == "the cat sat"
        assert results[0]["label"] == "cat"
        assert "score" in results[0]


class TestBM25Retriever:
    def test_index_and_retrieve(self):
        from sheepy.retrieval.retriever import BM25Retriever

        r = BM25Retriever()
        docs = ["machine learning is great", "deep learning rocks", "natural language processing"]
        labels = ["ml", "dl", "nlp"]
        r.index(docs, labels)

        results = r.retrieve("machine learning", k=2)
        assert len(results) == 2
        assert results[0]["label"] == "ml"


class TestRAGPipeline:
    def test_classify_majority_vote(self):
        from sheepy.retrieval.rag_pipeline import RAGPipeline

        class FakeRetriever:
            def retrieve(self, query, k=5):
                return [
                    {"text": "a", "label": "cat", "score": 0.9},
                    {"text": "b", "label": "cat", "score": 0.8},
                    {"text": "c", "label": "dog", "score": 0.7},
                ]

        pipe = RAGPipeline(FakeRetriever(), k=3)
        result = pipe.classify("test query")
        assert result["label"] == "cat"
        assert len(result["retrieved_docs"]) == 3

    def test_classify_batch(self):
        from sheepy.retrieval.rag_pipeline import RAGPipeline

        class FakeRetriever:
            def retrieve(self, query, k=5):
                return [{"text": "a", "label": "x", "score": 1.0}]

        pipe = RAGPipeline(FakeRetriever(), k=1)
        results = pipe.classify_batch(["q1", "q2"])
        assert len(results) == 2

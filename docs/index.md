# Sheepy

Experiment-tracking framework for text classification built on PyTorch Lightning, Weights & Biases, and Hugging Face Transformers.

See the [README](https://github.com/robmsylvester/sheepy) for a quick overview.

## Features

- **Transformer fine-tuning** — binary, multiclass, and multilabel classification
- **LLM classification** — zero-shot and few-shot via any LLM provider
- **Vector DB classification** — KNN and centroid classifiers with FAISS, ChromaDB, or HNSWlib
- **RAG classification** — retrieval-augmented pipelines with dense and BM25 retrieval
- **Rich W&B dashboards** — prediction tables, confusion matrices, PR/ROC curves, artifacts
- **Pydantic v2 config** — validated configuration from JSON, YAML, or TOML
- **Pluggable registry** — register custom models and data modules with decorators
- **Modern CLI** — `sheepy train`, `sheepy evaluate`, `sheepy predict`, `sheepy sweep`

## Installation

```bash
pip install sheepy          # core
pip install sheepy[all]     # all optional backends
pip install sheepy[llm]     # LLM support only
pip install sheepy[vector]  # vector DB support only
pip install sheepy[rag]     # retrieval support only
```

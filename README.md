# Sheepy

[![CI](https://github.com/robmsylvester/sheepy/actions/workflows/ci.yml/badge.svg)](https://github.com/robmsylvester/sheepy/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Experiment-tracking framework for text classification** built on [PyTorch Lightning](https://lightning.ai/), [Weights & Biases](https://wandb.ai/), and [Hugging Face Transformers](https://huggingface.co/transformers/) — with support for LLMs, vector databases, and retrieval-augmented classification.

## Features

| Capability | Module | Install |
|---|---|---|
| **Transformer fine-tuning** — binary, multiclass, multilabel | `sheepy.models` | `pip install sheepy` |
| **Pydantic v2 config** — JSON, YAML, TOML | `sheepy.config` | included |
| **Rich W&B dashboards** — tables, PR/ROC curves, artifacts | `sheepy.wandb_utils` | included |
| **LLM classification** — zero-shot / few-shot via any provider | `sheepy.llm` | `pip install sheepy[llm]` |
| **Vector DB classification** — KNN, centroid, FAISS/ChromaDB/HNSW | `sheepy.vector` | `pip install sheepy[vector]` |
| **RAG classification** — dense + BM25 retrieval pipelines | `sheepy.retrieval` | `pip install sheepy[rag]` |
| **Pluggable registry** — `@register_model` / `@register_data_module` | `sheepy.registry` | included |
| **CLI** — `sheepy train`, `sheepy evaluate`, `sheepy predict`, `sheepy sweep` | `sheepy.cli` | included |

## Quick Start

### Installation

```bash
# Core (transformers + Lightning + W&B)
pip install sheepy

# With all optional backends
pip install sheepy[all]

# Development
pip install -e ".[dev]"
```

### GPU / CUDA

The default `pip install torch` wheel from PyPI is often **CPU-only**, so `torch.cuda.is_available()` stays `False` even when you have an NVIDIA GPU. Sheepy’s `Experiment` class still configures Lightning correctly; it will use **`accelerator="cuda"`** and **`devices=1`** automatically once PyTorch can see CUDA.

1. Install a **CUDA-enabled** PyTorch build for your OS and driver from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (pick the same CUDA major version as your driver).
2. Reinstall or verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`.
3. On run, check the console line starting with `Lightning Trainer: accelerator=` (and `output/.../config.json` for `lightning_accelerator`, `compute_environment`).

Optional YAML overrides under `hparams`:

- `accelerator: cuda` (or `cpu`, `mps`) — force a backend.
- `devices: 2` — use two GPUs (DDP). Default when CUDA is visible is **`devices: 1`** (single GPU).

Apple Silicon: if `torch.backends.mps.is_available()`, sheepy uses **`accelerator="mps"`** by default when CUDA is not available.

### Train a classifier

```bash
# Via CLI
sheepy train --config examples/config/config_sms_spam.json --data-dir examples/datasets/spam

# Via Python
from sheepy import SheepyConfig, Experiment
from sheepy.models import TransformerClassifier
from sheepy.data_modules import BaseCSVDataModule

config = SheepyConfig.from_file("config.yaml")
flat = config.to_flat_dict()
flat["data_dir"] = "data/"
flat["output_dir"] = "output/"
flat = Experiment.validate_experiment_config(flat)

exp = Experiment(flat)
exp.prepare_trainer(BaseCSVDataModule, TransformerClassifier)
exp.train()
exp.test()
```

### LLM zero-shot classification

```python
from sheepy.llm import PromptClassifier

clf = PromptClassifier(
    model="gpt-4o-mini",
    labels=["positive", "negative", "neutral"],
)
result = clf.classify("This product is amazing!")
print(result["label"])  # "positive"
print(clf.usage_summary)
```

### Vector DB / KNN classification

```python
from sheepy.vector import EmbeddingClassifier, VectorIndex, KNNClassifier

embedder = EmbeddingClassifier(model_name="all-MiniLM-L6-v2")
index = VectorIndex(backend="faiss", dim=384)
knn = KNNClassifier(embedder, index, k=5)

knn.fit(train_texts, train_labels)
predictions = knn.predict(test_texts)
```

### RAG-augmented classification

```python
from sheepy.retrieval import DenseRetriever, RAGPipeline

retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
retriever.index(documents, labels)

pipeline = RAGPipeline(retriever, k=5)
result = pipeline.classify("What category is this?")
print(result["label"], result["retrieved_docs"])
```

### Custom models via registry

```python
import sheepy
from sheepy.models import BaseClassifier

@sheepy.register_model("my_classifier")
class MyClassifier(BaseClassifier):
    def _build_model(self):
        ...
    def forward(self, tokens, lengths):
        ...
    def configure_optimizers(self):
        ...
```

Then reference `"my_classifier"` in your config file.

## Configuration

Sheepy uses Pydantic v2 models for validated configuration. Configs can be JSON, YAML, or TOML:

```yaml
experiment:
  project_name: my_project
  model: transformer_classifier
  data_module: base_csv_data_module

hparams:
  encoder_model: bert-base-uncased
  batch_size: 32
  learning_rate: 3e-5
  num_labels: 2
  label: label
  text: text

metrics:
  multilabel: false
  tracked_metrics:
    - name: accuracy
      average: micro
    - name: f1
      average: macro

validation:
  metric: val/F1Score
  metric_goal: max

sweep:
  method: bayes
  metric:
    name: val/F1Score
    goal: maximize
  parameters:
    learning_rate:
      min: 1.0e-5
      max: 5.0e-4
```

## Architecture

```
sheepy/
├── cli.py                  # Typer CLI (train, evaluate, predict, sweep)
├── registry.py             # Pluggable model/data_module registry
├── config/                 # Pydantic v2 configuration models
├── common/                 # Logging, collation, DataFrame ops, S3, timestamps
├── nlp/                    # Tokenizer, LabelEncoder
├── metrics/                # torchmetrics v1 classification metrics
├── models/                 # BaseClassifier, TransformerClassifier, inference
├── data_modules/           # BaseDataModule, CSV, MultiLabel data modules
├── experiment/             # Experiment orchestrator, W&B sweep runner
├── wandb_utils/            # Tables, artifacts, PR/ROC curves, dashboard helpers
├── vector/                 # Embedding, VectorIndex, KNN, Centroid classifiers
├── retrieval/              # Dense/BM25 retrievers, RAG pipeline
└── llm/                    # LLM prompt-based classifier with cost tracking
```

## W&B Integration

Sheepy provides rich W&B logging out of the box:

- **Scalar metrics** — train/val/test loss, accuracy, F1, precision, recall, AUROC
- **Prediction tables** — interactive tables with text, true/predicted labels, confidence
- **Confusion matrices** — per-epoch, per-class for multilabel
- **PR and ROC curves** — via `wandb.plot`
- **Dataset artifacts** — versioned train/val/test splits
- **Model artifacts** — versioned checkpoints
- **Per-class metric panels** — auto-grouped by W&B
- **Experiment metadata** — git commit, config hash

## Running Tests

```bash
# All tests (fast)
pytest tests/ -m "not slow"

# Including slow tests (downloads models)
pytest tests/

# With coverage
pytest tests/ --cov=sheepy --cov-report=html
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

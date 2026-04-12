# Changelog

## [1.0.0] - 2026-04-11

### Breaking Changes
- Python 3.10+ required (dropped 3.8/3.9)
- Configuration now uses Pydantic v2 models instead of raw JSON + argparse Namespace
- Models now accept `config: dict` instead of `args: argparse.Namespace`
- `MulticlassTransformerClassifier` and `MultiLabelTransformerClassifier` are now thin wrappers around `TransformerClassifier` with `task=` parameter
- Removed `pytorch-nlp` dependency

### Added
- **`pyproject.toml`** — modern packaging with optional dependency groups (`[llm]`, `[vector]`, `[rag]`, `[dev]`)
- **`sheepy.config`** — Pydantic v2 config models with JSON/YAML/TOML support
- **`sheepy.cli`** — Typer-based CLI: `sheepy train`, `sheepy evaluate`, `sheepy predict`, `sheepy sweep`
- **`sheepy.vector`** — Embedding, VectorIndex (FAISS/ChromaDB/HNSW), KNN, Centroid classifiers
- **`sheepy.retrieval`** — Dense and BM25 retrievers, RAG pipeline for retrieval-augmented classification
- **`sheepy.llm`** — LLM prompt-based classifier (zero-shot, few-shot) with cost tracking via litellm
- **`sheepy.registry`** — `@register_model` / `@register_data_module` decorator API
- **`sheepy.wandb_utils`** — Prediction tables, dataset summaries, PR/ROC curves, artifacts, per-class metrics, experiment metadata logging
- **Structured logging** via structlog (replaces all `print()` statements)
- **GitHub Actions CI** — lint, typecheck, test across Python 3.10/3.11/3.12
- **Comprehensive test suite** — config, metrics, models, data modules, experiment, vector, retrieval, LLM, registry, W&B utils

### Fixed
- Confusion matrix `y_true`/`preds` argument order was swapped
- `MulticlassTransformerClassifier` had broken `_build_metrics()` and `validation_epoch_end()`
- `pandas.DataFrame.append()` replaced with `pd.concat()`
- All deprecated Lightning 1.x APIs migrated to Lightning 2.x
- torchmetrics v1 migration (`F1` -> `F1Score`, `task=` parameter, etc.)

### Removed
- `pytorch-nlp` dependency (replaced with inline implementation)
- `setup.py` as primary config (kept as thin shim)
- Poetry files under `examples/`

## [0.3.0] - 2022

- Initial public version with binary, multiclass, multilabel text classification
- PyTorch Lightning 1.5, Transformers 4.10, W&B 0.12

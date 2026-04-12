# Contributing to Sheepy

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/robmsylvester/sheepy.git
cd sheepy
pip install -e ".[dev,vector,rag]"
pre-commit install
```

## Code Style

- We use **ruff** for linting and formatting (replaces black, flake8, isort)
- Line length: 100 characters
- Type hints on all public functions
- Google-style docstrings

```bash
ruff check sheepy/ tests/
ruff format sheepy/ tests/
```

## Testing

```bash
# Run fast tests
pytest tests/ -m "not slow"

# Run all tests including model downloads
pytest tests/

# With coverage
pytest tests/ --cov=sheepy --cov-report=html
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass and linting is clean
4. Update documentation if adding new features
5. Open a PR using the template provided

## Adding New Models

1. Subclass `BaseClassifier` and implement `_build_model()`, `forward()`, and `configure_optimizers()`
2. Register with `@sheepy.register_model("my_model")`
3. Add tests in `tests/test_models.py`
4. Document in README

## Adding New Data Modules

1. Subclass `BaseDataModule` and implement `prepare_data()` and `setup()`
2. Register with `@sheepy.register_data_module("my_data")`
3. Add tests in `tests/test_data_modules.py`

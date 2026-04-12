"""Pluggable registry for models and data modules.

Register custom implementations with decorators::

    @sheepy.register_model("my_model")
    class MyModel(BaseClassifier):
        ...

    @sheepy.register_data_module("my_data")
    class MyData(BaseDataModule):
        ...
"""
from __future__ import annotations

from typing import Any

_MODEL_REGISTRY: dict[str, type] = {}
_DATA_MODULE_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class under *name*."""
    def decorator(cls: type) -> type:
        """Register *cls* in the model registry and return it unchanged."""
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_data_module(name: str):
    """Decorator to register a data module class under *name*."""
    def decorator(cls: type) -> type:
        """Register *cls* in the data-module registry and return it unchanged."""
        _DATA_MODULE_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> type | None:
    """Look up a registered model by name."""
    return _MODEL_REGISTRY.get(name)


def get_data_module(name: str) -> type | None:
    """Look up a registered data module by name."""
    return _DATA_MODULE_REGISTRY.get(name)


def list_models() -> list[str]:
    """Return all registered model names."""
    from sheepy.models import MODEL_MAPPING
    return sorted(set(list(MODEL_MAPPING.keys()) + list(_MODEL_REGISTRY.keys())))


def list_data_modules() -> list[str]:
    """Return all registered data module names."""
    from sheepy.data_modules import DATA_MODULE_MAPPING
    return sorted(set(list(DATA_MODULE_MAPPING.keys()) + list(_DATA_MODULE_REGISTRY.keys())))

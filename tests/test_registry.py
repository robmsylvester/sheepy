"""Tests for sheepy.registry."""
from sheepy.registry import (
    _DATA_MODULE_REGISTRY,
    _MODEL_REGISTRY,
    get_data_module,
    get_model,
    list_data_modules,
    list_models,
    register_data_module,
    register_model,
)


class TestRegistry:
    def test_register_and_get_model(self):
        @register_model("test_model_xyz")
        class TestModel:
            pass

        assert get_model("test_model_xyz") is TestModel
        # Cleanup
        _MODEL_REGISTRY.pop("test_model_xyz", None)

    def test_register_and_get_data_module(self):
        @register_data_module("test_dm_xyz")
        class TestDM:
            pass

        assert get_data_module("test_dm_xyz") is TestDM
        _DATA_MODULE_REGISTRY.pop("test_dm_xyz", None)

    def test_get_nonexistent_returns_none(self):
        assert get_model("nonexistent_xyz") is None
        assert get_data_module("nonexistent_xyz") is None

    def test_list_models_includes_builtins(self):
        models = list_models()
        assert "transformer_classifier" in models

    def test_list_data_modules_includes_builtins(self):
        dms = list_data_modules()
        assert "base_csv_data_module" in dms

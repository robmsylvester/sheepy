"""Sheepy — Experiment-tracking framework for text classification."""
__version__ = "1.0.0"

from sheepy.config.schema import SheepyConfig
from sheepy.experiment.base_experiment import Experiment
from sheepy.experiment.sweep_experiment import SweepExperiment
from sheepy.models.base_classifier import BaseClassifier
from sheepy.models.base_transformer_classifier import TransformerClassifier
from sheepy.registry import register_data_module, register_model

__all__ = [
    "__version__",
    "SheepyConfig",
    "Experiment",
    "SweepExperiment",
    "BaseClassifier",
    "TransformerClassifier",
    "register_model",
    "register_data_module",
]

"""Backward-compatible alias — multilabel behavior is now handled by
``BaseClassifier`` with ``task="multilabel"``.
"""
from sheepy.models.base_classifier import BaseClassifier

MultiLabelBaseClassifier = BaseClassifier

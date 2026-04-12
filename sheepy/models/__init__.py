from sheepy.models.base_classifier import BaseClassifier
from sheepy.models.base_transformer_classifier import TransformerClassifier
from sheepy.models.inference import SequenceClassificationModelRunner
from sheepy.models.multiclass_transformer_classifier import MulticlassTransformerClassifier
from sheepy.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier

MODEL_MAPPING: dict[str, type] = {
    "transformer_classifier": TransformerClassifier,
    "multiclass_transformer_classifier": MulticlassTransformerClassifier,
    "multilabel_transformer_classifier": MultiLabelTransformerClassifier,
}

__all__ = [
    "BaseClassifier",
    "TransformerClassifier",
    "MulticlassTransformerClassifier",
    "MultiLabelTransformerClassifier",
    "SequenceClassificationModelRunner",
    "MODEL_MAPPING",
]

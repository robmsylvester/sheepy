from sheepy.models.base_transformer_classifier import TransformerClassifier
from sheepy.models.multiclass_transformer_classifier import MulticlassTransformerClassifier
from sheepy.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier

MODEL_MAPPING = {
    "transformer_classifier": TransformerClassifier,
    "multiclass_transformer_classifier": MulticlassTransformerClassifier,
    "multilabel_transformer_classifier": MultiLabelTransformerClassifier,
}

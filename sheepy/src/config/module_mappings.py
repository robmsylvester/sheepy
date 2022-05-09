from sheepy.src.data_modules.base_csv_data_module import BaseCSVDataModule
from sheepy.src.data_modules.epilepsy_data_module import EpilepsyDataModule
from sheepy.src.data_modules.examples.semeval_sentiment_data_module import (
    SemEvalSentimentDataModule,
)
from sheepy.src.data_modules.examples.sms_spam_data_module import SmsSpamDataModule
from sheepy.src.data_modules.examples.toxic_comment_data_module import ToxicCommentDataModule
from sheepy.src.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule
from sheepy.src.models.base_transformer_classifier import TransformerClassifier
from sheepy.src.models.multiclass_transformer_classifier import MulticlassTransformerClassifier
from sheepy.src.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier

data_module_mapping = {
    'base_csv_data_module': BaseCSVDataModule,
    'toxic_comment_data_module': MultiLabelCSVDataModule,
    'sms_spam_data_module': SmsSpamDataModule,
    'toxic_comment_data_module': ToxicCommentDataModule,
    'semeval_data_module': SemEvalSentimentDataModule,
    'epilepsy_data_module': EpilepsyDataModule
}

model_mapping = {
    'transformer_classifier': TransformerClassifier,
    'multiclass_transformer_classifier': MulticlassTransformerClassifier,
    'multilabel_transformer_classifier': MultiLabelTransformerClassifier
}

from lib_ml_framework.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib_ml_framework.src.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule
from lib_ml_framework.src.data_modules.tweet_sentiment_data_module import TweetSentimentDataModule

from lib_ml_framework.src.data_modules.examples.sms_spam_data_module import SmsSpamDataModule

from lib_ml_framework.src.models.multilabel_augmented_transformer_classifier import MultiLabelAugmentedTransformerClassifier
from lib_ml_framework.src.models.base_transformer_classifier import TransformerClassifier
from lib_ml_framework.src.models.augmented_transformer_classifier import AugmentedTransformerClassifier
from lib_ml_framework.src.models.multiclass_transformer_classifier import MulticlassTransformerClassifier
from lib_ml_framework.src.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier


data_module_mapping = {
    'toxic_comment_data_module': MultiLabelCSVDataModule,
    'synthetic_tone_data_module': MultiLabelCSVDataModule,
    'sms_spam_data_module': SmsSpamDataModule,
    'base_csv_data_module': BaseCSVDataModule
}

model_mapping = {
    'augmented_transformer_classifier': AugmentedTransformerClassifier,
    'multilabel_augmented_transformer_classifier': MultiLabelAugmentedTransformerClassifier,
    'transformer_classifier': TransformerClassifier,
    'multiclass_transformer_classifier': MulticlassTransformerClassifier,
    'multilabel_transformer_classifier': MultiLabelTransformerClassifier
}

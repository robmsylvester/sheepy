from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule

from lib.src.data_modules.examples.sms_spam_data_module import SmsSpamDataModule
from lib.src.data_modules.examples.tweet_sentiment_data_module import TweetSentimentDataModule
from lib.src.data_modules.examples.semeval_sentiment_data_module import SemEvalSentimentDataModule

from lib.src.models.multilabel_augmented_transformer_classifier import MultiLabelAugmentedTransformerClassifier
from lib.src.models.base_transformer_classifier import TransformerClassifier
from lib.src.models.augmented_transformer_classifier import AugmentedTransformerClassifier
from lib.src.models.multiclass_transformer_classifier import MulticlassTransformerClassifier
from lib.src.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier


data_module_mapping = {
    'base_csv_data_module': BaseCSVDataModule,
    'toxic_comment_data_module': MultiLabelCSVDataModule,
    'sms_spam_data_module': SmsSpamDataModule,
    'tweet_sentiment_data_module': TweetSentimentDataModule,
    'semeval_data_module': SemEvalSentimentDataModule
}

model_mapping = {
    'augmented_transformer_classifier': AugmentedTransformerClassifier,
    'multilabel_augmented_transformer_classifier': MultiLabelAugmentedTransformerClassifier,
    'transformer_classifier': TransformerClassifier,
    'multiclass_transformer_classifier': MulticlassTransformerClassifier,
    'multilabel_transformer_classifier': MultiLabelTransformerClassifier
}

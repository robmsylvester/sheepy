from semeval_sentiment_data_module import SemEvalSentimentDataModule
from sms_spam_data_module import SmsSpamDataModule
from toxic_comment_data_module import ToxicCommentDataModule

from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule
from sheepy.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule

DATA_MODULE_MAPPING = {
    'base_csv_data_module': BaseCSVDataModule,
    'toxic_comment_data_module': MultiLabelCSVDataModule,
    'sms_spam_data_module': SmsSpamDataModule,
    'toxic_comment_data_module': ToxicCommentDataModule,
    'semeval_data_module': SemEvalSentimentDataModule,
}

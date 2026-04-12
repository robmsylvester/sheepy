from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule
from sheepy.data_modules.base_data_module import BaseDataModule
from sheepy.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule

DATA_MODULE_MAPPING: dict[str, type] = {
    "base_csv_data_module": BaseCSVDataModule,
    "multi_label_csv_data_module": MultiLabelCSVDataModule,
}

__all__ = [
    "BaseDataModule",
    "BaseCSVDataModule",
    "MultiLabelCSVDataModule",
    "DATA_MODULE_MAPPING",
]

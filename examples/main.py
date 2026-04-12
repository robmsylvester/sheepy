"""Example entry point demonstrating sheepy usage with the legacy CLI style.

For the modern CLI, use: ``sheepy train --config config.json``
"""
from __future__ import annotations

import json
import sys

import torch

from sheepy.config.schema import SheepyConfig
from sheepy.data_modules import DATA_MODULE_MAPPING
from sheepy.experiment.base_experiment import Experiment
from sheepy.experiment.sweep_experiment import SweepExperiment
from sheepy.models import MODEL_MAPPING

# Example-specific data modules
try:
    from sms_spam_data_module import SmsSpamDataModule
    DATA_MODULE_MAPPING["sms_spam_data_module"] = SmsSpamDataModule
except ImportError:
    pass

try:
    from toxic_comment_data_module import ToxicCommentDataModule
    DATA_MODULE_MAPPING["toxic_comment_data_module"] = ToxicCommentDataModule
except ImportError:
    pass

try:
    from semeval_sentiment_data_module import SemEvalSentimentDataModule
    DATA_MODULE_MAPPING["semeval_data_module"] = SemEvalSentimentDataModule
except ImportError:
    pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--time", type=str, default="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate_batch_file", type=str, default=None)
    parser.add_argument("--evaluate_live", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_key", type=str, default=None)
    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)

    args = parser.parse_args()

    cfg = SheepyConfig.from_file(args.config)
    config = cfg.to_flat_dict()

    config["evaluate"] = args.evaluate
    config["evaluate_live"] = args.evaluate_live
    config["evaluate_batch_file"] = args.evaluate_batch_file
    config["tune"] = args.tune
    config["fp16"] = args.fp16
    config["time"] = args.time
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.output_key:
        config["output_key"] = args.output_key
    if args.pretrained_dir:
        config["pretrained_dir"] = args.pretrained_dir
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    dm_key = config.get("data_module", "base_csv_data_module")
    model_key = config.get("model", "transformer_classifier")
    data_module_cls = DATA_MODULE_MAPPING.get(dm_key)
    model_cls = MODEL_MAPPING.get(model_key)

    if data_module_cls is None or model_cls is None:
        print(f"Unknown data_module='{dm_key}' or model='{model_key}'")
        sys.exit(1)

    config = Experiment.validate_experiment_config(config)

    if args.tune:
        experiment = SweepExperiment(config, data_module_cls, model_cls)
        experiment.tune()
    else:
        experiment = Experiment(config)
        if args.evaluate:
            experiment.prepare_evaluator(data_module_cls, model_cls)
            if args.evaluate_live:
                while True:
                    experiment.predict_live()
            else:
                experiment.predict_batch()
        else:
            experiment.prepare_trainer(data_module_cls, model_cls)
            experiment.train()
            experiment.test()


if __name__ == "__main__":
    main()

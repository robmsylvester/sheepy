import argparse
import torch
import time
import os
import json
from lib.src.experiment.base_experiment import Experiment
from lib.src.experiment.sweep_experiment import SweepExperiment
from lib.src.config.module_mappings import data_module_mapping, model_mapping


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # The config file below specifies values for all arguments listed in this function
    # TODO - some these can be moved to a config, such as fp16
    parser.add_argument("--config", required=True, type=str,
                        help="Location of experiment config profile JSON.")
    parser.add_argument("--time", type=str, required=True,
                        help="current time str, populated by shell most likely, in format YYYY_MM_dd__hh_mm_ss_UTC")
    parser.add_argument("--fp16", action="store_true",
                        help="Enables 16-bit floating-point precision during training on GPU's if available")
    parser.add_argument("--data_format", type=str, default="csv",
                        help="Data format of input file/s, must be either csv or json")
    parser.add_argument("--evaluate", action="store_true",
                        help="Runs the model in batch evaluation mode on a text file instead of the train/val/test mode.")
    parser.add_argument("--evaluate_batch_file", type=str, default=None,
                        help="Location of the text file with which to run batch evaluation.")
    parser.add_argument("--evaluate_live", action="store_true",
                        help="Runs the model in raw input evaluation mode from the shell instead of the train/val/test mode or batch evaluation mode")
    parser.add_argument("--tune", action="store_true",
                        help="Runs the model in tune/sweep mode as per the sweep argument of the config file.")
    parser.add_argument("--data_output", type=str, default=None,
                        help="Runs the model in evaluation mode instead of the train/val/test mode")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Name of the output column or dict key. Used only in evaluation mode")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Name of the directory with pretrained model. Used only in evaluation mode")

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
        try:
            data_module_key = config["experiment"]["data_module"]
            model_key = config["experiment"]["model"]
            data_module = data_module_mapping[data_module_key]
            model = model_mapping[model_key]
        except KeyError:
            raise KeyError(
                "You must pass both a 'data_module' and 'model' argument to the config file under the 'experiment' key object, and these must be mapped in the module_mappings.py file")

    if args.tune and args.evaluate:
        raise ValueError(
            "You cannot run the model with both tune and evaluate mode turned on. Pick one or the other, or neither if you want to just train.")

    parser = data_module.add_model_specific_args(parser)
    parser = model.add_model_specific_args(parser)
    parser = Experiment.add_model_specific_args(parser)

    # Use the environment to verify a few additional arguments
    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    args.precision = 16 if args.fp16 and args.n_gpu > 0 else 32

    #TODO - move this all to experiment class
    # Verify model output directories
    if args.evaluate:
        
        if not args.evaluate_live:
            if not args.evaluate_batch_file:
                raise ValueError("When running in batch evaluation mode, you must provide a text file to the evaluate_batch_file argument")
            if not os.path.exists(args.evaluate_batch_file):
                raise ValueError("Cannot find evaluate_batch_file in file system located at {}".format(args.evaluate_batch_file))

        if args.experiment_name is None or args.output_key is None:
            raise ValueError(
                "You must pass a pretrained experiment name and the output column name or dict key as output_key.")

        if args.pretrained_dir is None:
            args.pretrained_dir = os.path.join(
                args.output_dir, args.project_name, args.experiment_name)

        args.output_dir = os.path.join(
            args.pretrained_dir, 'eval')

        if args.data_output is None:
            args.data_output = os.path.join(
                args.data_dir, 'eval', args.experiment_name)

        if not os.path.exists(args.data_output):
            os.makedirs(args.data_output)
    else:
        if args.experiment_name is None:
            args.experiment_name = "{}bit_{}_v{}".format(
                args.precision, args.time, args.version)

        args.output_dir = os.path.join(
            args.output_dir, args.project_name, args.experiment_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load up an experiment and make sure config is okay for it. We always call config
    if args.tune:
        experiment = SweepExperiment(args, data_module, model)
        experiment.tune()
    else:
        experiment = Experiment(args)
        if args.evaluate:
            experiment.prepare_evaluator(data_module, model)
            if args.evaluate_live:
                while True:
                    experiment.predict_live()
            else:
                experiment.predict_batch()
        else:
            experiment.prepare_trainer(data_module, model)
            experiment.train()
            experiment.test()


if __name__ == '__main__':
    main()

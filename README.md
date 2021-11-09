# Transformer Classification Framework

## Purpose

This repository consists of the skeleton for what is needed to run experiments on PyTorch Lightning and Weights and Biases. It
was built to help standardize the process of training and evaluating models as well as provide improved transparency with
respect to model performance and custom success metrics.

There are a few base classes used for ETL and modeling that can help with reusable components.

This framework is made primarily for NLP models for text classification using transformers, supporting multi-class and multi-label experiments as well.

## Installation and Preparation
1. Clone repository
2. Create virtual environment `virtualenv -p python3.6 .venv`
3. Install local requirements
   `pip install --upgrade pip`
   `pip install -r requirements.txt`
4. Run setup script (optional)
   `python setup.py install`
6. Login to Weights and Biases. Note that you may have to create a free account.
   `wandb login`

## Usage Demo Scripts
1. `run_experiment.sh` will run train/prediction depending on shell args. This is for running NLP experiments
2. You'll notice that a .env file is referenced in the shell script. Use this to store some constants for paths (s3 or local) from your own data.

## Check Out
See `src/base_data_module.py` for an introduction to the ETL classes.
See `src/main.py()` for a high-level of full execution.
See `src/demo_predict.py()` for a quick example of inference.

## FAQ
1. If you receive a runtime error that looks like this while training:
RuntimeError: [Errno 2] No such file or directory: `/tmp/some_random_stuff.graph.json`, 
then it is related to a runtime bug under the hood with Weights and Biases and you can just run your code again and it should work. This happens from time to time.

2. If you receive an error related to the transformers library being unable to download one of the pretrained models, this has happened to me from time to time as well. They are stored on AWS, and for me it has always worked when running a second time. Let me know if you have some issues with this.
# Transformer Text Classification Framework

## Purpose

This library consists of the skeleton for what is needed to run experiments on PyTorch Lightning and Weights and Biases. It
was built to help standardize the process of training and evaluating models as well as provide improved transparency with
respect to model performance and custom success metrics.

There are a few base classes used for ETL and modeling.

## Installation and Preparation
1. Clone repository
2. Create virtual environment `virtualenv -p python3.7 .venv`
(This is what version of Python I used to develop this. Later versions might give you problems)
3. Install local requirements
   `pip install --upgrade pip`
   `pip install -r requirements.txt`
4. Run setup script
   `python3.7 setup.py install`
5. Install system requirements for ONNX (https://github.com/Microsoft/onnxruntime#system-requirements)
6. Login to (and possibly create) Weights and Biases Account
   `wandb login`

## Usage Demo Scripts
1. `run_train_experiment.sh` will run train/tune experiments depending on shell args
2. `run_eval_experiment.sh` will run prediction experiments depending on shell args
3. `run_tests.sh` will run tests.

## Check Out
See `src/base_data_module.py` for an introduction to the ETL classes.
See `src/main.py()` for a high-level of full execution.

## FAQ
If you receive a runtime error that looks like this while training:
RuntimeError: [Errno 2] No such file or directory: `/tmp/some_random_stuff.graph.json`, then it is related to a runtime bug under the hood with Weights and Biases and you can just run your code again and it should work. This happens from time to time.

If you are installing this on Ubuntu/Debian and are installing Python3.7 from source, you'll need to grab the bz2 headers first. Before the build steps, run: `sudo apt-get install libbz2-dev`. Make sure you also have the standard build tools


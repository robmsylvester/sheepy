# Sheepy

## Purpose

This library consists of the skeleton for what is needed to run experiments on PyTorch Lightning and Weights and Biases,
specifically targeted at using Huggingface transformers to perform classification. It includes examples for CSV data modules
written for binary, multiclass, and multilabel classification, as well as multiple modular models that you should be able to
inherit from for your task. It also contains code for the Shap value package to easily interpret results as well as torch metrics to be able to output what you want in weights and biases. There is example code for creating additional matplotlib visualizations included in the form of a confusion matrix.

## Installation and Preparation
1. Clone repository
1. Create virtual environment `python -m venv .venv`. Use Python3.8 or Python3.9. 
(This is what version of Python I used to develop this. Later versions might give you problems)
1. Install local requirements
   `pip install --upgrade pip`
   `pip install -r requirements.txt`
1. Activate the environment: `source .venv/bin/activate`
1. Run setup script
   `python setup.py install`
1. Install system requirements for ONNX (https://github.com/Microsoft/onnxruntime#system-requirements)
1. Login to (and possibly create) Weights and Biases Account
   `wandb login`

## Usage Demo Scripts
There are demo scripts that can be used to showcase the framework.

### Binary Classification
1. `./run_spam_train_experiment.sh` will run train/tune experiments depending on shell args for the binary email spam classification task
2. `./run_spam_eval_experiment.sh` will run prediction experiments, either in batches or live via the command line, on your trained model.

### Multiclass Classification
1. `./run_semeval_train_experiment.sh` will run train/tune experiments depending on shell args for the three-class tweet sentiment classification task
2. `./run_semeval_eval_experiment.sh` will run prediction experiments, either in batches or live via the command line, on your trained model.

### Multilabel Classification
1. `./run_toxic_train_experiment.sh` will run train/tune experiments depending on shell args for the multilabel toxic comment classification task.
2. `./run_toxic_eval_experiment.sh` will run prediction experiments, either in batches or live via the command line, on your trained model.

### Tests
`./run_tests.sh` will run tests.

## Weights and Biases
You'll need a (free) weights and biases account to visualize metrics that are tracked in your experiment. When running
the shell scripts, you'll be prompted to create/login to it.

## Important Classes
### Experiments
The experiment class houses the highest level abstraction. It consists of
1. Lightning data module for your experiment
1. Lightning model for your experiment
1. Lightning trainer object used to train/infer
1. Experiment config JSON (see below)
1. Label encoder/decoders
1. Transformer tokenizers and model wrappers
1. Weights and biases logger
1. Code for reporting metrics to the logger
1. Callbacks in training (early stopping, learning rate decay, etc.)
1. Stdout loggers
1. Code to save/load modules
1. Hyperparameter tuning (SweepExperiment class)

### Experiment Configs
Many of the arguments for the data modules and models are shared between different tasks, such as which metrics you want to
track, or which labels you want to track. In `sheepy/src/config` you will find example configurations for all the samples as
well as a thorough explanation of what they all mean in `config_guide.json`.
There are four keys in the top-level of this config:
1. "experiment" - unique identifiers that is used to create directories and save/load models. Also where you point to your data modules and lightning modules
1. "metrics" - list of pytorch metrics that you care about tracking.
1. "hparams" - houses all the (hyper)parameters that you want access to in your modules. this are designed to be module-specific
1. "validation" - helps to track which metrics you aim to optimize for, as well as help with certain callbacks (early stopping, etc.)
1. "sweep" - holds necessary parameters for W&B sweep experiments.

## Modes
The logic for which mode your experiment runs in can be found easily in `main.py`, but to summarize it in natural language...

### Training
This mode is used for a classic train/val/test split, which leans heavily on pytorch lightning. It makes use of train, val, and test data loaders.

### Evaluation
This mode is used for prediction. It makes use of predict data loaders in pytorch lightning. The included examples support either a batch evaluation
whose results are written to disk, or a live evaluation whose results are sent back to the console in a REPL shell.

### Tuning
This mode is used for hyperparameter tuning. It makes use of weights and biases and the config sweep JSON to run a specific optimization technique and report back the best results. Currently the bugs are still being worked out here, but this should be available soon.

## Using The Framework For New Experiments/Datasets
To use the framework yourself on your own dataset, you'll need to.
1. Create a config JSON modeled after one of the examples
1. Create a data module. It will likely inherit from the BaseDataModule, or maybe the BaseCSVDataModule. These are called like normal
lightning data modules, so anything you want to override in prepare_data(), setup(), or any of the other lifecycle hooks you will need to
write here.
1. Any command line args you want to add can be done with the add_model_specific_args class method as seen in the other data module examples.
1. Optionally create a model (lightning module) Base transformer classifiers are already written, but you might have some extra engineered features that you wish to concatenate using the augmented_transformer_classifier, for example.
1. Make sure the name of your data module and model are referenced in some sort of runner script. `examples/main.py` is a runner script that you can use by default for your code. You can place the data module mapping in DATA_MODULE_MAPPING. You can place the model mapping in `models/__init__.py`
1. Create a shell script that calls the experiment class pointing at whatever arguments you need to specify. See the examples and, again, reference `main.py`
1. You'll want to register your code changes py reinstalling the setup script: `python setup.py install`

## FAQ
1. If you receive a runtime error that looks like this while training:
RuntimeError: [Errno 2] No such file or directory: `/tmp/some_random_stuff.graph.json`, then it is related to a runtime bug under the hood with Weights and Biases and you can just run your code again and it should work. This happens from time to time.

2. If you are installing this on Ubuntu/Debian and are installing Python3.7 from source, you'll possibly need to grab the bz2 headers first, especially
if you have not been using python much on that filesystem. Before the build steps, run: `sudo apt-get install libbz2-dev`. Make sure you also have the standard build tools.

3. If you run into memory errors, try dropping the batch size or using a 128-output vector model
instead of a 768-output model.

4. Check CUDA and Nvidia driver compatibility with respect to not only your own system but also pytorch. If you are developing on a newer version of CUDA, you probably need a pytorch nightly build, and you may need to explicitly install that version in your virtual environment

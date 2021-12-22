from setuptools import setup, find_packages
import os
import sys
sys.path.append(os.path.dirname(__file__))

__version__ = "0.0.4"

setup(
    name='transformer_classification_framework',
    version=__version__,
    description='Experiment-tracking wrapper around PyTorch Lightning, Weights & Biases, Pandas, and Huggingface Trasformers that aims at rapid experimentation for common text classification tasks using transformers',
    author='Rob Sylvester',
    author_email='robmsylvester@gmail.co.com',
    python_requires='==3.7',
    packages=find_packages()
)

from setuptools import setup, find_packages
import os
import sys
sys.path.append(os.path.dirname(__file__))

__version__ = "0.1.4"

setup(
    name='transformer_classification_framework',
    version=__version__,
    description='Experiment-tracking wrapper around PyTorch Lightning, Weights & Biases, Pandas, and Huggingface Trasformers that aims at rapid experimentation for common text classification tasks using transformers',
    author='Rob Sylvester',
    author_email='robmsylvester@gmail.com',
    python_requires='>=3.8,<3.10',
    install_requires=[
        "pandas>=1.1.1",
        "torch>=1.6.0",
        "transformers==4.10.2",
        "pytorch_lightning==1.5.8",
        "pytorch-nlp==0.5.0",
        "wandb==0.12.6",
        "numpy>=1.20.0",
        "scikit-learn==0.22.1",
        "plotly==4.10.0",
        "pytest~=6.2.2",
        "Pillow~=8.1.0",
        "tqdm~=4.56.0",
        "setuptools~=53.0.0",
        "torchmetrics==0.6.0"
    ],
    packages=find_packages()
)
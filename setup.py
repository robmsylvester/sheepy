import os
import re
import sys

from setuptools import find_packages, setup

sys.path.append(os.path.dirname(__file__))
PACKAGE_NAME = "sheepy"
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_INIT_FILE = os.path.join(CURRENT_DIR, PACKAGE_NAME, "__init__.py")
with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as fp:
    README = fp.read()
with open(PACKAGE_INIT_FILE, encoding="utf-8") as fp:
    VERSION = re.search('__version__ = "([^"]+)"', fp.read()).group(1)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    long_description=README,
    description="Experiment-tracking wrapper around PyTorch Lightning, Weights & Biases, Pandas, and Huggingface Trasformers that aims at rapid experimentation for common text classification tasks using transformers",
    author="Rob Sylvester",
    author_email="robmsylvester@gmail.com",
    python_requires=">=3.8,<3.10",
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
        "torchmetrics==0.6.0",
        "rich>=12.4.1",
    ],
    packages=find_packages(),
)

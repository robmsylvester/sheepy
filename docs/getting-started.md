# Getting Started

## Installation

```bash
pip install sheepy
```

## Your First Experiment

### 1. Create a config file

```yaml
# config.yaml
experiment:
  project_name: my_project
  model: transformer_classifier
  data_module: base_csv_data_module

hparams:
  encoder_model: bert-base-uncased
  batch_size: 16
  learning_rate: 3e-5
  num_labels: 2
  label: label
  text: text

metrics:
  tracked_metrics:
    - name: accuracy
    - name: f1
      average: macro

validation:
  metric: val/F1Score
  metric_goal: max
```

### 2. Train via CLI

```bash
sheepy train --config config.yaml --data-dir data/
```

### 3. Or train via Python

```python
from sheepy import SheepyConfig, Experiment
from sheepy.models import TransformerClassifier
from sheepy.data_modules import BaseCSVDataModule

config = SheepyConfig.from_file("config.yaml")
flat = config.to_flat_dict()
flat["data_dir"] = "data/"
flat["output_dir"] = "output/"
flat = Experiment.validate_experiment_config(flat)

exp = Experiment(flat)
exp.prepare_trainer(BaseCSVDataModule, TransformerClassifier)
exp.train()
exp.test()
```

## Using from Another Project

```bash
pip install sheepy
```

Then in your code:

```python
from sheepy import Experiment, TransformerClassifier, SheepyConfig
from sheepy.vector import KNNClassifier, EmbeddingClassifier, VectorIndex
from sheepy.llm import PromptClassifier
```

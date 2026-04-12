# Configuration

Sheepy uses Pydantic v2 models for validated configuration. All configs can be JSON, YAML, or TOML.

## Config Sections

### `experiment`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `project_name` | str | `"default_project"` | W&B project name |
| `description` | str | `""` | Experiment description |
| `data_module` | str | `"base_csv_data_module"` | Registered data module name |
| `model` | str | `"transformer_classifier"` | Registered model name |
| `version` | str/int | `"0"` | Experiment version |

### `hparams`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `encoder_model` | str | `"bert-base-uncased"` | HuggingFace model name |
| `batch_size` | int | `16` | Training batch size |
| `learning_rate` | float | `3e-5` | Optimizer learning rate |
| `encoder_learning_rate` | float | `1e-5` | Encoder-specific learning rate |
| `num_labels` | int | `2` | Number of classification labels |
| `label` | str/list | `"label"` | Label column name(s) |
| `text` | str | `"text"` | Text column name |
| `num_epochs` | int | `10` | Maximum training epochs |
| `train_ratio` | float | `0.7` | Train split ratio |
| `validation_ratio` | float | `0.15` | Validation split ratio |
| `loader_workers` | int | `4` | `DataLoader` worker processes |
| `accelerator` | str / null | `null` | Lightning accelerator override (`cuda`, `cpu`, `mps`). `null` = pick CUDA, else MPS, else CPU. |
| `devices` | int / str / null | `null` | Lightning `devices`. With CUDA and `null`, defaults to **1** GPU. Use `2` or `"auto"` for multi-GPU (DDP). |

### `metrics`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `multilabel` | bool | `false` | Whether this is a multilabel task |
| `tracked_metrics` | list | `[]` | List of metric definitions |

Each metric definition has `name` (e.g. `"accuracy"`, `"f1"`, `"precision"`, `"recall"`, `"auroc"`) and optional `average` (`"macro"`, `"micro"`, `"weighted"`).

### `validation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `metric` | str | `"val/F1Score"` | Metric to monitor for checkpointing |
| `metric_goal` | str | `"max"` | `"max"` or `"min"` |
| `check_val_every_n_epoch` | int | `1` | Validation frequency |

### `sweep`

Standard W&B sweep configuration. See [W&B docs](https://docs.wandb.ai/guides/sweeps).

## Loading Config

```python
from sheepy.config import SheepyConfig

# From file (JSON, YAML, or TOML)
config = SheepyConfig.from_file("config.yaml")

# From dict
config = SheepyConfig.model_validate({"hparams": {"batch_size": 32}})

# To flat dict for Experiment
flat = config.to_flat_dict()
```

"""Sheepy CLI — ``sheepy train``, ``sheepy evaluate``, ``sheepy predict``, ``sheepy sweep``."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="sheepy",
    help="Experiment-tracking framework for text classification.",
    add_completion=False,
)


def _load_config(config_path: str) -> dict:
    """Load and flatten a Sheepy config file into a dict for CLI commands."""
    from sheepy.config.schema import SheepyConfig
    cfg = SheepyConfig.from_file(config_path)
    return cfg.to_flat_dict()


def _resolve_classes(config: dict):
    """Resolve ``data_module`` and ``model`` keys to concrete classes from the registry or built-ins."""
    from sheepy.data_modules import DATA_MODULE_MAPPING
    from sheepy.models import MODEL_MAPPING
    from sheepy.registry import get_data_module, get_model

    dm_key = config.get("data_module", "base_csv_data_module")
    model_key = config.get("model", "transformer_classifier")

    data_module_cls = get_data_module(dm_key) or DATA_MODULE_MAPPING.get(dm_key)
    model_cls = get_model(model_key) or MODEL_MAPPING.get(model_key)

    if data_module_cls is None:
        raise typer.BadParameter(f"Unknown data_module: '{dm_key}'")
    if model_cls is None:
        raise typer.BadParameter(f"Unknown model: '{model_key}'")

    return data_module_cls, model_cls


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment config file."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory."),
    fp16: bool = typer.Option(False, "--fp16", help="Enable mixed precision."),
    experiment_name: Optional[str] = typer.Option(None, "--name", "-n", help="Experiment name."),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Data directory."),
) -> None:
    """Train a text classification model."""
    from sheepy.experiment.base_experiment import Experiment

    cfg = _load_config(config)
    cfg["evaluate"] = False
    cfg["tune"] = False
    if output_dir:
        cfg["output_dir"] = output_dir
    if fp16:
        cfg["fp16"] = True
    if experiment_name:
        cfg["experiment_name"] = experiment_name
    if data_dir:
        cfg["data_dir"] = data_dir

    cfg = Experiment.validate_experiment_config(cfg)
    data_module_cls, model_cls = _resolve_classes(cfg)

    experiment = Experiment(cfg)
    experiment.prepare_trainer(data_module_cls, model_cls)
    experiment.train()
    experiment.test()
    typer.echo("Training complete.")


@app.command()
def evaluate(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment config file."),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Pretrained model directory."),
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="Text file for batch eval."),
    live: bool = typer.Option(False, "--live", help="Interactive prediction mode."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Evaluate a trained model on data."""
    from sheepy.experiment.base_experiment import Experiment

    cfg = _load_config(config)
    cfg["evaluate"] = True
    cfg["evaluate_live"] = live
    if checkpoint:
        cfg["pretrained_dir"] = checkpoint
    if input_file:
        cfg["evaluate_batch_file"] = input_file
    if output_dir:
        cfg["output_dir"] = output_dir

    cfg = Experiment.validate_experiment_config(cfg)
    data_module_cls, model_cls = _resolve_classes(cfg)

    experiment = Experiment(cfg, evaluation=True)
    experiment.prepare_evaluator(data_module_cls, model_cls)

    if live:
        typer.echo("Live prediction mode. Press Ctrl+C to exit.")
        while True:
            try:
                experiment.predict_live()
            except KeyboardInterrupt:
                break
    else:
        experiment.predict_batch()
    typer.echo("Evaluation complete.")


@app.command()
def predict(
    model_dir: str = typer.Option(..., "--model-dir", "-m", help="Saved HF model directory."),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to classify."),
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="File of texts, one per line."),
    min_prob: float = typer.Option(0.0, "--min-prob", help="Minimum probability threshold."),
) -> None:
    """Run standalone inference on text."""
    from sheepy.models.inference import SequenceClassificationModelRunner

    runner = SequenceClassificationModelRunner(model_dir)

    if text:
        results = runner.predict(text, min_prob=min_prob)
        for r in results:
            typer.echo(f"  {r['label_name']}: {r['prob']:.4f}")
    elif input_file:
        with open(input_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            typer.echo(f"\n> {line}")
            for r in runner.predict(line, min_prob=min_prob):
                typer.echo(f"  {r['label_name']}: {r['prob']:.4f}")
    else:
        typer.echo("Provide --text or --input.")
        raise typer.Exit(1)


@app.command()
def sweep(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment config file."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Run a W&B hyperparameter sweep."""
    from sheepy.experiment.base_experiment import Experiment
    from sheepy.experiment.sweep_experiment import SweepExperiment

    cfg = _load_config(config)
    cfg["evaluate"] = False
    cfg["tune"] = True
    if output_dir:
        cfg["output_dir"] = output_dir

    cfg = Experiment.validate_experiment_config(cfg)
    data_module_cls, model_cls = _resolve_classes(cfg)

    experiment = SweepExperiment(cfg, data_module_cls, model_cls)
    experiment.tune()
    typer.echo("Sweep complete.")


@app.command()
def version() -> None:
    """Print the sheepy version."""
    from sheepy import __version__
    typer.echo(f"sheepy {__version__}")


if __name__ == "__main__":
    app()

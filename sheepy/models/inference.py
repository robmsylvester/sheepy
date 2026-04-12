"""Standalone inference runner for saved HuggingFace models."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassificationModelRunner:
    """Load a saved sheepy model directory and run predictions."""

    def __init__(self, model_dir: str | Path) -> None:
        """Load ``config.json``, weights, and tokenizer from a saved sheepy export directory."""
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    @torch.no_grad()
    def predict(self, text: str | list[str], min_prob: float = 0.0) -> list[dict]:
        """Run inference on text input.

        Args:
            text: Single string or list of strings.
            min_prob: Filter out predictions below this probability.

        Returns:
            List of dicts with ``label_name`` and ``prob`` keys.
        """
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for batch_probs in probs:
            df = pd.DataFrame({
                "prob": batch_probs,
                "label_name": self.config.get("custom_args", {}).get(
                    "label", [f"label_{i}" for i in range(len(batch_probs))]
                ),
            })
            df = df.sort_values("prob", ascending=False)
            results.extend(df[df["prob"] > min_prob].to_dict(orient="records"))
        return results

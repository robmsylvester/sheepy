import json
import os

import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassificationModelRunner:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def predict(self, text, min_prob=0.0):
        inputs = self.tokenizer(text, return_tensors="pt")
        logits = self.model(**inputs)[0]
        probs = torch.sigmoid(logits)
        probs = probs.detach().cpu().numpy()

        df = pd.DataFrame(
            {
                "prob": probs[0],
                "label_name": self.config["label"],
            }
        )
        df = df.sort_values("prob", ascending=False)
        return df[df["prob"] > min_prob].to_dict(orient="records")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassificationModelRunner:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def predict(self, text, min_prob=0.0):
        inputs = self.tokenizer(text)
        logits = self.model(**inputs)[0]
        probs = torch.sigmoid(logits)
        probs = probs.detach().cpu().numpy()

        df = pd.DataFrame(
            {
                "prob": probs[0],
                "label_name": self.label_names,
            }
        )
        df = df.sort_values("prob", ascending=False)
        return df[df["prob"] > min_prob].to_dict(orient="records")

import os
from typing import Dict, List

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, layer_properties: dict):
        """See FullyConnectedClassifier documentation for the specifications of the input config object"""
        super().__init__()
        self.dense = nn.Linear(
            layer_properties["input_size"],
            layer_properties["output_size"]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(layer_properties["dropout_p"])

    def forward(self, input_layer: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.dense(input_layer)
        hidden_layer = self.relu(hidden_layer)
        hidden_layer = self.dropout(hidden_layer)
        return hidden_layer


class FullyConnectedClassifier(nn.Module):
    def __init__(self, layers: List[Dict]):
        """The config object is a list of dictionaries that specify layer properties. Each element in this list is a dictionary that has:
            input_size: int,
            output_size: int,
            dropout: float, the probability of zero'ing activation during training (just to hammer that in...this is the probability of turning OFF the activation)

            The last element in the list should have an output_size = the number of desired logits in your classifier
            The output_size of element n must be equal to the input size of element n+1
            Dropout will be ignored on the final element of the list

            For now, all activations are rectified linear units.
        """
        super().__init__()
        dense_layers = []
        for dense_layer_idx in range(len(layers) - 1):
            dense_layers.append(DenseLayer(layers[dense_layer_idx]))
        # TODO - probably become nn.Sequential
        self.classifier_layers = nn.ModuleList(dense_layers)
        self.classifier = nn.Linear(
            layers[-1]["input_size"],
            layers[-1]["output_size"]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.classifier_layers:
            hidden_states = layer(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_path, "classification_head.bin"))

"""
Neural network models used in the AIA-FL paper.

Paper: "Attribute Inference Attacks for Federated Regression Tasks"
Model: Single hidden-layer neural network with 128 neurons and ReLU activation.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SequentialNet(nn.Module):
    """
    Feedforward neural network with configurable hidden layers.

    Default configuration matches the paper:
        - 1 hidden layer with 128 neurons
        - ReLU activation
        - Output dimension 1 (regression)

    Args:
        input_dimension: Number of input features.
        output_dimension: Number of output neurons (1 for regression).
        hidden_layers: List of hidden layer sizes. Defaults to [128].
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int = 1,
        hidden_layers: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128]

        if len(hidden_layers) > 0:
            layers: List[nn.Module] = [
                nn.Linear(input_dimension, hidden_layers[0]),
                nn.ReLU(),
            ]
            for i in range(1, len(hidden_layers)):
                layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()]
            layers.append(nn.Linear(hidden_layers[-1], output_dimension))
        else:
            layers = [nn.Linear(input_dimension, output_dimension)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearModel(nn.Module):
    """
    Linear regression model (no hidden layers, no activation).
    Used for the least-squares experiments in the paper.
    """

    def __init__(self, input_dimension: int, output_dimension: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def get_model(
    model_type: str,
    input_dimension: int,
    output_dimension: int = 1,
    hidden_layers: Optional[List[int]] = None,
) -> nn.Module:
    """
    Factory function for model creation.

    Args:
        model_type: 'neural_network' or 'linear'.
        input_dimension: Number of input features.
        output_dimension: Number of outputs.
        hidden_layers: Hidden layer sizes for neural network.

    Returns:
        Initialized PyTorch model.
    """
    if model_type == "neural_network":
        hidden = hidden_layers if hidden_layers is not None else [128]
        return SequentialNet(input_dimension, output_dimension, hidden)
    elif model_type == "linear":
        return LinearModel(input_dimension, output_dimension)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'neural_network' or 'linear'.")


def get_model_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Return list of parameter tensors (used for Flower's ndarrays interface)."""
    return [param.data.clone() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors."""
    for param, new_param in zip(model.parameters(), parameters):
        param.data.copy_(new_param)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Return all model parameters as a single flat tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a single flat tensor."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset: offset + numel].view(param.shape))
        offset += numel


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

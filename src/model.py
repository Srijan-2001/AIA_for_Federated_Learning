"""
model.py — Neural network models for FL classification experiments.

Architectures:
  - SimpleCNN    : lightweight 2-conv CNN for MNIST / FashionMNIST (1-channel)
  - CIFARNet     : deeper CNN for CIFAR-10 / CIFAR-100 (3-channel)
  - SequentialNet: feedforward NN for regression (AIA paper experiments)
  - LinearModel  : linear regression (AIA paper least-squares experiments)
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Two-conv CNN for 1-channel images (MNIST, FashionMNIST).
    Input: (B, 1, 28, 28)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CIFARNet(nn.Module):
    """
    3-conv block CNN for 3-channel images (CIFAR-10 / CIFAR-100).
    Input: (B, 3, 32, 32)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Regression models (AIA paper)
# ---------------------------------------------------------------------------

class SequentialNet(nn.Module):
    """
    Feedforward neural network — paper default: 1 hidden layer, 128 neurons, ReLU.
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

        if hidden_layers:
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
    """Linear regression model (no hidden layers)."""

    def __init__(self, input_dimension: int, output_dimension: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(
    dataset_name: str,
    num_classes: Optional[int] = None,
    # Regression kwargs
    model_type: Optional[str] = None,
    input_dimension: Optional[int] = None,
    output_dimension: int = 1,
    hidden_layers: Optional[List[int]] = None,
) -> nn.Module:
    """
    Return the appropriate model for a given dataset.

    For classification datasets ('mnist', 'fmnist', 'cifar10', 'cifar100'):
        Uses SimpleCNN or CIFARNet.

    For regression tasks (model_type='neural_network'/'linear'):
        Uses SequentialNet or LinearModel.
    """
    name = (dataset_name or "").lower()

    # Regression mode
    if model_type in ("neural_network", "linear"):
        if input_dimension is None:
            raise ValueError("input_dimension required for regression model")
        if model_type == "neural_network":
            return SequentialNet(input_dimension, output_dimension, hidden_layers or [128])
        return LinearModel(input_dimension, output_dimension)

    # Classification mode
    if name in ("mnist", "fmnist"):
        nc = num_classes or 10
        return SimpleCNN(nc)
    elif name in ("cifar10", "cifar100"):
        nc = num_classes or (100 if name == "cifar100" else 10)
        return CIFARNet(nc)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Pass model_type='neural_network'/'linear' with input_dimension for regression tasks."
        )


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view(p.shape))
        offset += n


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Return approximate model size in megabytes."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)

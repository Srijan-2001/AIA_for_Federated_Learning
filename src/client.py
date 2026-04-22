"""
client.py — Flower FL client with standardized FedAvg local training.

Instruction-compliant settings (Section 4.1):
  - Optimizer: SGD with momentum=0.9
  - Learning rate: 0.01
  - Local epochs: 5 per communication round
  - Batch size: 32
  - Loss: CrossEntropyLoss (classification) / MSELoss (regression)
  - Seeds: 42

The client also saves (global, local) model checkpoints each round to support
passive eavesdrop attacks (paper Algorithm 2).
"""

import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    Status, Code,
)

logger = logging.getLogger(__name__)


class FedAvgClient(fl.client.Client):
    """
    FedAvg client — standardized per course instructions.

    Local training (each round):
      for epoch in range(local_epochs):         # default 5
        for batch in train_loader:              # batch_size 32
          loss = criterion(model(x), y)
          SGD step (lr=0.01, momentum=0.9)

    Args:
        client_id: Unique client string ID.
        model: PyTorch model.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        local_epochs: Local training epochs per round (default 5).
        learning_rate: SGD learning rate (default 0.01).
        momentum: SGD momentum (default 0.9).
        task: 'classification' or 'regression'.
        device: Torch device.
        checkpoint_dir: Directory to save (global, local) checkpoints.
        save_checkpoints: Whether to save checkpoints (needed for AIA).
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        task: str = "classification",
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        save_checkpoints: bool = True,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.task = task
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints

        # Loss function per instruction
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # SGD with momentum per instruction
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )

        self.round_counter = 0

        if checkpoint_dir:
            os.makedirs(os.path.join(checkpoint_dir, client_id), exist_ok=True)

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def fit(self, ins: FitIns) -> FitRes:
        """Receive global params, train locally, return updated params."""
        self.round_counter += 1
        rnd = str(self.round_counter)

        # Load global model
        global_params = parameters_to_ndarrays(ins.parameters)
        self._set_parameters(global_params)

        # Save global checkpoint (before local update) — for AIA eavesdropping
        if self.save_checkpoints and self.checkpoint_dir:
            gpath = os.path.join(self.checkpoint_dir, self.client_id, f"global_{rnd}.pt")
            torch.save({"model_state_dict": self.model.state_dict()}, gpath)

        # Local training
        train_loss = self._local_train()

        # Save local checkpoint (after local update)
        if self.save_checkpoints and self.checkpoint_dir:
            lpath = os.path.join(self.checkpoint_dir, self.client_id, f"local_{rnd}.pt")
            torch.save({"model_state_dict": self.model.state_dict()}, lpath)

        n_train = len(self.train_loader.dataset)
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(self._get_parameters()),
            num_examples=n_train,
            metrics={
                "client_id": self.client_id,
                "round": self.round_counter,
                "train_loss": float(train_loss),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate global model on local test set."""
        self._set_parameters(parameters_to_ndarrays(ins.parameters))
        loss, accuracy = self._evaluate()
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"accuracy": float(accuracy), "client_id": self.client_id},
        )

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------

    def _local_train(self) -> float:
        """Run local_epochs of SGD. Returns average training loss."""
        self.model.train()
        total_loss, n = 0.0, 0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                if self.task == "classification":
                    loss = self.criterion(out, y.long())
                else:
                    loss = self.criterion(out.squeeze(), y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(y)
                n += len(y)
        return total_loss / max(n, 1)

    def _evaluate(self) -> Tuple[float, float]:
        """Compute loss and accuracy on the test loader."""
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                if self.task == "classification":
                    loss = self.criterion(out, y.long())
                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                else:
                    loss = self.criterion(out.squeeze(), y.float())
                total_loss += loss.item() * len(y)
                n += len(y)
        avg_loss = total_loss / max(n, 1)
        accuracy = correct / max(n, 1) if self.task == "classification" else 0.0
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_parameters(self) -> List[np.ndarray]:
        return [p.data.cpu().numpy() for p in self.model.parameters()]

    def _set_parameters(self, params: List[np.ndarray]) -> None:
        for p, new_p in zip(self.model.parameters(), params):
            p.data.copy_(torch.from_numpy(new_p).to(self.device))

"""
Flower FL Client — FedAvg local training.

Each client:
  1. Receives global model parameters from server.
  2. Performs local_epochs of SGD on its private dataset.
  3. Returns updated parameters + metadata to the server.

The client also stores (global_params, local_params) pairs for use
by the passive adversary's eavesdrop attack.
"""

import logging
import os
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Flower imports
import flwr as fl
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status,
    Code,
)

from models import get_model, get_flat_params, set_flat_params, count_parameters
from utils import binary_accuracy, mse, save_checkpoint

logger = logging.getLogger(__name__)


class FedAvgClient(fl.client.Client):
    """
    Federated Averaging client implementing the FedAvg local update rule.

    Matches Algorithm 4 of the paper:
      for each local epoch e from 1 to E:
        for batch b in B:
          θ_c(k+1) ← θ_c(k) − η × (1/|b|) Σ_{x∈b} ∇ℓ(θ_c(k), x)

    Args:
        client_id: Unique string identifier.
        model: PyTorch neural network.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        local_epochs: Number of local training epochs (E in the paper).
        learning_rate: SGD learning rate η.
        device: Torch device.
        checkpoint_dir: Directory to save model checkpoints for attack analysis.
        save_checkpoints: Whether to save (global, local) model checkpoints each round.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_epochs: int = 1,
        learning_rate: float = 5e-7,
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
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints

        self.criterion = nn.MSELoss()
        # momentum=0.9 matches the paper's baseline SGD configuration (Item 15)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        # Track round counter (used for checkpoint file names)
        self.round_counter = 0

        # Stored metadata: round_id → {global_path, local_path}
        self.messages_metadata: Dict[str, Dict[str, str]] = {"global": {}, "local": {}}

        if checkpoint_dir:
            os.makedirs(os.path.join(checkpoint_dir, client_id), exist_ok=True)

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def fit(self, ins: FitIns) -> FitRes:
        """
        Receive global model, run local training, return updated model.
        """
        self.round_counter += 1
        round_id = str(self.round_counter)

        # 1. Load global model parameters
        global_params = parameters_to_ndarrays(ins.parameters)
        self._set_parameters(global_params)

        # 2. Save global checkpoint BEFORE local update
        if self.save_checkpoints and self.checkpoint_dir:
            global_path = os.path.join(
                self.checkpoint_dir, self.client_id, f"global_{round_id}.pt"
            )
            save_checkpoint(self.model, global_path)
            self.messages_metadata["global"][round_id] = global_path

        global_flat = get_flat_params(self.model).clone()

        # 3. Local training (FedAvg local update, Algorithm 4)
        self._local_train()

        local_flat = get_flat_params(self.model).clone()

        # 4. Save local checkpoint AFTER local update
        if self.save_checkpoints and self.checkpoint_dir:
            local_path = os.path.join(
                self.checkpoint_dir, self.client_id, f"local_{round_id}.pt"
            )
            save_checkpoint(self.model, local_path)
            self.messages_metadata["local"][round_id] = local_path

        # 5. Return updated parameters
        updated_params = self._get_parameters()
        n_train = len(self.train_loader.dataset)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=n_train,
            metrics={
                "client_id": self.client_id,
                "round": self.round_counter,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the global model on the client's test set."""
        global_params = parameters_to_ndarrays(ins.parameters)
        self._set_parameters(global_params)

        loss, accuracy = self._evaluate_loader(self.test_loader)
        n_test = len(self.test_loader.dataset)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=n_test,
            metrics={"mse_loss": float(loss), "client_id": self.client_id},
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _local_train(self) -> None:
        """Run local_epochs of SGD (FedAvg local update rule)."""
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float()

                self.optimizer.zero_grad()
                preds = self.model(x).squeeze()
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        """Compute MSE loss on a DataLoader. Returns (avg_mse, avg_mse).
        The second value is sent to the server as the 'mse_loss' metric.
        """
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                preds = self.model(x).squeeze()
                total_loss += nn.MSELoss(reduction="sum")(preds, y).item()
                n += len(y)
        avg_loss = total_loss / max(n, 1)
        # For regression, return MSE as both "loss" and "metric"
        return avg_loss, avg_loss

    # ------------------------------------------------------------------
    # Parameter helpers (interface with Flower ndarrays)
    # ------------------------------------------------------------------

    def _get_parameters(self) -> List[np.ndarray]:
        return [p.data.cpu().numpy() for p in self.model.parameters()]

    def _set_parameters(self, params: List[np.ndarray]) -> None:
        for p, new_p in zip(self.model.parameters(), params):
            p.data.copy_(torch.from_numpy(new_p).to(self.device))

    # ------------------------------------------------------------------
    # Helpers for the attack pipeline
    # ------------------------------------------------------------------

    def get_flat_global_params_history(self) -> List[torch.Tensor]:
        """Return list of flat global model param tensors across rounds."""
        result = []
        for round_id in sorted(self.messages_metadata["global"].keys(), key=int):
            path = self.messages_metadata["global"][round_id]
            m = self.model.__class__(
                *[p.shape[0] for p in self.model.parameters()][:1]
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            tmp_model = deepcopy(self.model)
            tmp_model.load_state_dict(ckpt["model_state_dict"])
            result.append(get_flat_params(tmp_model))
        return result

    def get_flat_local_params_history(self) -> List[torch.Tensor]:
        result = []
        for round_id in sorted(self.messages_metadata["local"].keys(), key=int):
            path = self.messages_metadata["local"][round_id]
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            tmp_model = deepcopy(self.model)
            tmp_model.load_state_dict(ckpt["model_state_dict"])
            result.append(get_flat_params(tmp_model))
        return result

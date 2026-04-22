"""
server.py — Flower FedAvg server strategy with mandatory metrics tracking.

Tracks per-round (per instructions Section 5):
  - Global test accuracy
  - Global test loss
  - Convergence round (first round accuracy >= threshold, default 80%)
  - Communication cost (MB)

Also extends FedAvgWithAIA from fl_server.py for AIA experiments.
"""

import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    Metrics,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

logger = logging.getLogger(__name__)


class StandardFedAvg(fl.server.strategy.Strategy):
    """
    FedAvg strategy compliant with course instructions (Section 4.1):
      - client_fraction = 0.5 (50% sampled per round)
      - local_epochs = 5
      - batch_size = 32
      - SGD, lr=0.01, momentum=0.9
      - CrossEntropyLoss
      - Seeds: 42

    Tracks mandatory metrics:
      - global test accuracy each round
      - global test loss each round
      - convergence_round (first round accuracy >= convergence_threshold)
      - communication_cost_mb (cumulative)

    Args:
        model_init_fn: Callable() → fresh nn.Module.
        initial_parameters: Starting Flower Parameters.
        test_loader: Global held-out test DataLoader.
        num_classes: Number of classes (for accuracy computation).
        fraction_fit: Client fraction per round (default 0.5 per instructions).
        convergence_threshold: Accuracy threshold for convergence round (default 0.80).
        device: Torch device.
    """

    def __init__(
        self,
        model_init_fn,
        initial_parameters: Parameters,
        test_loader,
        num_classes: int = 10,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        convergence_threshold: float = 0.80,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_init_fn = model_init_fn
        self.initial_parameters = initial_parameters
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.convergence_threshold = convergence_threshold
        self.device = device

        self._current_params: List[np.ndarray] = parameters_to_ndarrays(initial_parameters)
        self._round = 0

        # Mandatory metrics history
        self.history_accuracy: List[Tuple[int, float]] = []
        self.history_loss: List[Tuple[int, float]] = []
        self.convergence_round: Optional[int] = None
        self.communication_cost_mb: float = 0.0

    # ------------------------------------------------------------------
    # Flower Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._round = server_round
        n_fit, min_n = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=n_fit, min_num_clients=min_n)
        ins = FitIns(parameters=parameters, config={"round": server_round})
        return [(c, ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(r.parameters), r.num_examples)
            for _, r in results
        ]
        aggregated = aggregate(weights_results)
        self._current_params = aggregated

        # Track communication cost: sum of all param bytes × num clients × 2 (up+down)
        model_size = sum(a.nbytes for a in aggregated) / (1024 ** 2)
        self.communication_cost_mb += model_size * len(results) * 2

        return ndarrays_to_parameters(aggregated), {"round": server_round}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        n_eval, min_n = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=n_eval, min_num_clients=min_n)
        ins = EvaluateIns(parameters=parameters, config={})
        return [(c, ins) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        loss = weighted_loss_avg([(r.num_examples, r.loss) for _, r in results])
        return loss, {"round": server_round}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate on the global test set every round."""
        model = self.model_init_fn().to(self.device)
        for p, arr in zip(model.parameters(), parameters_to_ndarrays(parameters)):
            p.data.copy_(torch.from_numpy(arr).to(self.device))

        loss, accuracy = self._evaluate_global(model)
        self.history_accuracy.append((server_round, accuracy))
        self.history_loss.append((server_round, loss))

        # Track convergence round
        if self.convergence_round is None and accuracy >= self.convergence_threshold:
            self.convergence_round = server_round
            logger.info(
                f"Convergence at round {server_round} "
                f"(accuracy={accuracy:.4f} >= {self.convergence_threshold})"
            )

        logger.info(
            f"Round {server_round:3d} | Global test acc={accuracy:.4f} | "
            f"loss={loss:.4f} | comm_cost={self.communication_cost_mb:.2f} MB"
        )
        return loss, {"accuracy": accuracy, "round": server_round}

    # ------------------------------------------------------------------
    # Global evaluation helper
    # ------------------------------------------------------------------

    def _evaluate_global(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on the global test loader."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                total_loss += criterion(out, y.long()).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                n += len(y)
        return total_loss / max(n, 1), correct / max(n, 1)

    def get_metrics_summary(self) -> Dict:
        """Return a summary of all tracked mandatory metrics."""
        return {
            "accuracy_history": self.history_accuracy,
            "loss_history": self.history_loss,
            "convergence_round": self.convergence_round,
            "communication_cost_mb": round(self.communication_cost_mb, 4),
            "final_accuracy": self.history_accuracy[-1][1] if self.history_accuracy else None,
            "final_loss": self.history_loss[-1][1] if self.history_loss else None,
        }

    # ------------------------------------------------------------------
    # Client sampling helpers
    # ------------------------------------------------------------------

    def num_fit_clients(self, num_available: int) -> Tuple[int, int]:
        n = max(int(num_available * self.fraction_fit), self.min_fit_clients)
        return n, self.min_fit_clients

    def num_evaluation_clients(self, num_available: int) -> Tuple[int, int]:
        n = max(int(num_available * self.fraction_evaluate), self.min_evaluate_clients)
        return n, self.min_evaluate_clients

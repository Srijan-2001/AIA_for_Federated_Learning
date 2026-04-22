"""
Model-based Attribute Inference Attack (AIA).

Paper: "Attribute Inference Attacks for Federated Regression Tasks", Diana et al. 2025

Implements:
  - ModelBasedAIA  (Eq. 3 from the paper): given a model θ, infer sc(i) by:
        argmin_{sc(i)} ℓ(θ, (x^p_c(i), sc(i), y^p_c(i)))
    For binary sensitive attributes this reduces to checking which value
    (0 or 1) yields the smaller loss — an exact closed-form solution.

  - LinearModelReconstructionAttack (Algorithm 2): reconstruct optimal local
    model for least-squares regression under a passive adversary.

  - ActiveModelReconstructionAttack (Algorithm 3): active adversary uses
    Adam emulation to drive the client toward its local optimum.

Theoretical backing:
  Proposition 1: AIA accuracy ≥ 1 − 4E_c / θ[s]^2
  The lower the local MSE (more overfitting), the higher the AIA accuracy.
"""

import logging
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core model-based AIA (Eq. 3)
# ---------------------------------------------------------------------------

class ModelBasedAIA:
    """
    Model-based Attribute Inference Attack.

    For a binary sensitive attribute: enumerate both possible values {0, 1}
    and assign the value with lower model loss (exactly solving Eq. 3).

    Args:
        model: The target model θ (typically the reconstructed local model).
        dataset: Client's dataset with public features + labels.
        sensitive_attr_id: Column index of the sensitive attribute.
        criterion: Loss function (MSELoss for regression).
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        sensitive_attr_id: int,
        criterion: nn.Module,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.sensitive_attr_id = sensitive_attr_id
        self.criterion = criterion.to(device)
        self.device = device

        # Load all samples at once
        from torch.utils.data._utils.collate import default_collate
        if len(dataset) == 0:
            raise ValueError(
                "ModelBasedAIA received an empty dataset. "
                "The targeted client's training split has 0 samples. "
                "This usually means the heterogeneous split produced an empty shard — "
                "try a different --targeted_client index, reduce --heterogeneity, or "
                "increase the dataset size."
            )
        all_samples = [dataset[i] for i in range(len(dataset))]
        self.features, self.labels = default_collate(all_samples)
        self.features = self.features.to(device).float()
        self.labels = self.labels.to(device).float()
        self.n_samples = len(self.features)

        # Bounds of the sensitive attribute in the dataset
        self.s_min = self.features[:, sensitive_attr_id].min().item()
        self.s_max = self.features[:, sensitive_attr_id].max().item()

        self.predicted_sensitive = self.features[:, sensitive_attr_id].clone()

    def execute_attack(self) -> torch.Tensor:
        """
        Execute the model-based AIA.

        For each sample i, pick the sensitive attribute value in {s_min, s_max}
        that minimises ℓ(θ, (x^p_c(i), sc(i), y^p_c(i))).

        Returns:
            Tensor of predicted sensitive attribute values (shape: [n_samples]).
        """
        self.model.eval()
        predicted = torch.zeros(self.n_samples, device=self.device)

        with torch.no_grad():
            for i in range(self.n_samples):
                x = self.features[i].clone()
                y = self.labels[i]

                # Try s_min
                x_0 = x.clone()
                x_0[self.sensitive_attr_id] = self.s_min
                loss_0 = self.criterion(self.model(x_0.unsqueeze(0)).squeeze(), y)

                # Try s_max
                x_1 = x.clone()
                x_1[self.sensitive_attr_id] = self.s_max
                loss_1 = self.criterion(self.model(x_1.unsqueeze(0)).squeeze(), y)

                predicted[i] = self.s_min if loss_0 <= loss_1 else self.s_max

        self.predicted_sensitive = predicted
        return predicted

    def evaluate_attack(self) -> float:
        """
        Compute AIA accuracy: fraction of correctly inferred sensitive attributes.

        Returns:
            Accuracy in [0, 1].
        """
        true_attr = self.features[:, self.sensitive_attr_id].clone()
        correct = (torch.round(self.predicted_sensitive) == torch.round(true_attr)).float()
        return correct.mean().item()

    def evaluate_attack_mse(self) -> float:
        """
        Compute reconstruction MSE between predicted and true sensitive attribute values.

        Returns MSE (mean squared error) between the raw predicted sensitive attribute
        values and the true values — this is the Section 5.2 reconstruction MSE metric.

        Returns:
            MSE in [0, ∞).
        """
        true_attr = self.features[:, self.sensitive_attr_id].clone()
        mse_val = torch.nn.functional.mse_loss(
            self.predicted_sensitive.float(), true_attr.float()
        ).item()
        return mse_val


# ---------------------------------------------------------------------------
# Linear Model Reconstruction Attack — passive adversary (Algorithm 2)
# ---------------------------------------------------------------------------

class LinearModelReconstructionAttack:
    """
    Reconstruct the client's optimal local model from eavesdropped messages.
    Only valid for least-squares (linear) regression.

    Algorithm 2 from the paper:
      Given nc pairs (θ^{t_i}, θ^{t_i}_c(K)) [global, local] across rounds:
        Θ_in  = [θ^{t_1}  ... θ^{t_nc}]^T        shape (nc, d)
        Θ_out = [(θ^{t_1} - θ^{t_1}_c)^T | 1 ]   shape (nc, d+1)
        (V̂, θ̂*_c) = last row of (Θ_out^T Θ_out)^† Θ_out^T Θ_in

    Theorem 1: reconstruction error → 0 as nc → ∞.
    Proposition 2: exact recovery when B = S_c (full-batch gradient).

    Args:
        global_params_list: List of global model parameter flat tensors [nc × d].
        local_params_list:  List of local model parameter flat tensors [nc × d].
    """

    def __init__(
        self,
        global_params_list: List[torch.Tensor],
        local_params_list: List[torch.Tensor],
    ):
        assert len(global_params_list) == len(local_params_list), \
            "global and local parameter lists must have equal length"
        assert len(global_params_list) >= 2, \
            "Need at least 2 message pairs to reconstruct"

        self.global_params = global_params_list   # [(d,), ...]
        self.local_params = local_params_list     # [(d,), ...]
        self.nc = len(global_params_list)
        self.d = global_params_list[0].shape[0]

    def reconstruct(self) -> torch.Tensor:
        """
        Run Algorithm 2 and return the reconstructed optimal local model.

        Returns:
            Flat parameter tensor of shape (d,).
        """
        # Θ_in: shape (nc, d)
        Theta_in = torch.stack(self.global_params, dim=0).float()          # (nc, d)

        # Θ_out: columns = (θ_global - θ_local) + bias column of ones
        # shape (nc, d+1)
        diffs = [g - l for g, l in zip(self.global_params, self.local_params)]
        diffs_mat = torch.stack(diffs, dim=0).float()                       # (nc, d)
        ones_col = torch.ones(self.nc, 1, dtype=torch.float32)
        Theta_out = torch.cat([diffs_mat, ones_col], dim=1)                 # (nc, d+1)

        # Least-squares: (Θ_out^T Θ_out)^† Θ_out^T Θ_in
        # Result shape: (d+1, d) — last row is θ*_c
        TtT = Theta_out.T @ Theta_out                                        # (d+1, d+1)
        TtT_inv = torch.linalg.pinv(TtT)                                     # (d+1, d+1)
        solution = TtT_inv @ Theta_out.T @ Theta_in                          # (d+1, d)

        # Last row is the reconstructed optimal local model (shape d)
        theta_star_c = solution[-1, :]
        return theta_star_c


# ---------------------------------------------------------------------------
# Active Model Reconstruction Attack (Algorithm 3)
# ---------------------------------------------------------------------------

class ActiveModelReconstructionAttack:
    """
    Reconstruct the client's optimal local model using an active (malicious) adversary.

    Algorithm 3 from the paper:
      The adversary repeatedly sends back the same model to the client (instead
      of the averaged global model), causing the client to converge toward its
      local optimum. Adam emulation is used to track the client's gradient
      information and accelerate convergence.

    This class maintains the adversary's internal state and provides:
      - get_malicious_model(): the model to inject at each active round
      - update(): update internal state given client's returned model

    Args:
        initial_model_params: Flat parameter tensor from the latest client update.
        lr: Adam learning rate. Default from paper hyperparameter search.
        beta1: Adam β₁.
        beta2: Adam β₂.
        eps: Adam ε.
        device: Torch device.
    """

    def __init__(
        self,
        initial_model_params: torch.Tensor,
        lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        device: str = "cpu",
    ):
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # θ^a_c: adversary's current model (starts from last client update)
        self.theta_a = initial_model_params.clone().to(device).float()

        # Adam moment vectors
        self.m = torch.zeros_like(self.theta_a)   # 1st moment
        self.v = torch.zeros_like(self.theta_a)   # 2nd moment
        self.t = 0                                  # step counter

    def get_malicious_model_params(self) -> torch.Tensor:
        """Return the current adversarial model parameters to inject."""
        return self.theta_a.clone()

    def update(self, client_returned_params: torch.Tensor) -> None:
        """
        Update the adversary's model using Adam on the pseudo-gradient.

        The pseudo-gradient = θ^a_c − θ_c (Algorithm 3, line 5).

        Args:
            client_returned_params: Flat parameters of the model returned by the client.
        """
        self.t += 1
        client_params = client_returned_params.clone().to(self.device).float()

        # Pseudo-gradient: difference between adversary's injected model and client's update
        pseudo_grad = self.theta_a - client_params   # shape (d,)

        # Adam update (Kingma & Ba 2015, Algorithm 1)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * pseudo_grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (pseudo_grad ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        self.theta_a = self.theta_a - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    @property
    def reconstructed_params(self) -> torch.Tensor:
        """Return the current estimate of the client's optimal local model."""
        return self.theta_a.clone()

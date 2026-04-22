"""
Gradient-based Attribute Inference Attack baseline.

Paper: "Attribute Inference Attacks for Federated Regression Tasks", Diana et al. 2025
Baseline: Lyu & Chen (2021) / Chen et al. (2022)

The attack identifies sensitive attribute values that produce virtual gradients
closely matching the client's pseudo-gradients in terms of cosine similarity:

    argmax_{s_c(i)} Σ_{t ∈ T} CosSim(
        ∂ℓ(θ^t, {(x^p_c(i), s_c(i), y^p_c(i))}) / ∂θ^t,
        θ^t − θ^t_c
    )

For binary sensitive attributes, this is solved with Gumbel-Softmax
reparameterisation (Jang, Gu & Poole 2017).

Implementation follows Section 2.3 of the paper.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def gumbel_softmax_binary(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Gumbel-Softmax for a binary variable.

    Args:
        logits: Tensor of shape (n_samples, 1).
        tau: Temperature.
        hard: If True, return hard (0/1) samples in forward, soft in backward.
        generator: Optional RNG.

    Returns:
        Tensor of shape (n_samples,) with values in {0, 1} (hard) or (0,1) (soft).
    """
    # FIX: torch.rand_like() does not accept a 'generator' keyword argument.
    # Use torch.rand() with explicit shape, device and dtype to match rand_like
    # behaviour while still accepting the generator for reproducibility.
    gumbel_noise = -torch.log(
        -torch.log(
            torch.rand(
                logits.shape,
                generator=generator,
                device=logits.device,
                dtype=logits.dtype,
            ) + 1e-20
        ) + 1e-20
    )
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau).squeeze(-1)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # Straight-through estimator
        y_out = y_hard - y_soft.detach() + y_soft
    else:
        y_out = y_soft

    return y_out


class GradientBasedAIA:
    """
    Gradient-based Attribute Inference Attack.

    Implements the cosine-similarity gradient matching attack from
    Lyu & Chen (2021) and Chen et al. (2022), extended to support
    both passive and active adversaries.

    For best performance:
      - Passive: use all inspected rounds T ⊆ T_c
      - Active: additionally use attack rounds T^a_c

    Args:
        model_init_fn: Callable returning a fresh nn.Module.
        dataset: Client's dataset.
        sensitive_attr_id: Index of the sensitive attribute column.
        criterion: Loss function (MSELoss for regression).
        device: Torch device.
        learning_rate: SGD learning rate for logit optimisation.
        gumbel_tau: Gumbel-Softmax temperature.
        seed: Random seed.
    """

    def __init__(
        self,
        model_init_fn,
        dataset: Dataset,
        sensitive_attr_id: int,
        criterion: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e4,
        gumbel_tau: float = 1.0,
        seed: int = 42,
    ):
        self.model_init_fn = model_init_fn
        self.dataset = dataset
        self.sensitive_attr_id = sensitive_attr_id
        self.criterion = criterion.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.gumbel_tau = gumbel_tau

        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)

        # Load all samples
        from torch.utils.data._utils.collate import default_collate
        all_samples = [dataset[i] for i in range(len(dataset))]
        self.true_features, self.true_labels = default_collate(all_samples)
        self.true_features = self.true_features.to(device).float()
        self.true_labels = self.true_labels.to(device).float()
        self.n_samples = len(self.true_features)

        # Sensitive attribute bounds
        self.s_min = self.true_features[:, sensitive_attr_id].min().item()
        self.s_max = self.true_features[:, sensitive_attr_id].max().item()

        # Logits for the sensitive attribute (to be optimised)
        # Shape: (n_samples, 1)
        logits = torch.randn(
            self.n_samples, 1, device=device, generator=self.torch_rng
        )
        self.sensitive_logits = logits.clone().detach().requires_grad_(True)

        self.optimizer = torch.optim.SGD([self.sensitive_logits], lr=self.learning_rate)

        # Container for (pseudo-gradient, global model) pairs collected across rounds
        self._round_data: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []

    # ------------------------------------------------------------------
    # Round data collection
    # ------------------------------------------------------------------

    def add_round(
        self,
        global_params: List[torch.Tensor],
        local_params: List[torch.Tensor],
    ) -> None:
        """
        Register one communication round.

        Args:
            global_params: Global model parameter tensors (flat list).
            local_params: Client's returned parameter tensors (flat list).
        """
        # Pseudo-gradient: θ^t - θ^t_c(K)  (flat vector)
        pseudo_grad = torch.cat([
            (g - l).view(-1)
            for g, l in zip(global_params, local_params)
        ]).to(self.device).detach()

        self._round_data.append((pseudo_grad, [p.to(self.device) for p in global_params]))

    def clear_rounds(self) -> None:
        self._round_data.clear()

    # ------------------------------------------------------------------
    # AIA iteration
    # ------------------------------------------------------------------

    def _get_sensitive_attribute(self, hard: bool = True) -> torch.Tensor:
        """
        Sample sensitive attribute from current logits using Gumbel-Softmax.

        Returns:
            Tensor of shape (n_samples,) with values approximately in [s_min, s_max].
        """
        s_raw = gumbel_softmax_binary(
            self.sensitive_logits, tau=self.gumbel_tau, hard=hard,
            generator=self.torch_rng,
        )
        # Linearly scale to [s_min, s_max]
        s = s_raw * (self.s_max - self.s_min) + self.s_min
        return s

    def _compute_virtual_gradient(
        self,
        model: nn.Module,
        predicted_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the virtual gradient of the loss w.r.t. model parameters.

        Returns:
            Flat gradient tensor.
        """
        model.zero_grad()
        preds = model(predicted_features).squeeze()
        loss = self.criterion(preds, self.true_labels)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return torch.cat([g.view(-1) for g in grads])

    def _iteration(self) -> Tuple[float, float]:
        """
        One optimisation step: minimise Σ_t (1 - CosSim(virtual_grad, pseudo_grad)).

        Returns:
            (loss_value, current_attack_accuracy)
        """
        self.optimizer.zero_grad()

        s = self._get_sensitive_attribute(hard=True)
        predicted_features = self.true_features.clone()
        predicted_features[:, self.sensitive_attr_id] = s

        total_loss = torch.tensor(0.0, device=self.device)

        for pseudo_grad, global_params in self._round_data:
            model = self.model_init_fn().to(self.device)
            # Set model to global params for this round
            offset = 0
            for p in model.parameters():
                n = p.numel()
                p.data.copy_(
                    torch.cat([gp.view(-1) for gp in global_params])[offset: offset + n].view(p.shape)
                )
                offset += n

            virtual_grad = self._compute_virtual_gradient(model, predicted_features)
            cos_sim = F.cosine_similarity(virtual_grad, pseudo_grad, dim=0)
            total_loss = total_loss + (1.0 - cos_sim)

        # Backpropagate through logits
        grad_logits = torch.autograd.grad(total_loss, self.sensitive_logits, retain_graph=True)[0]
        self.sensitive_logits.grad = grad_logits
        self.optimizer.step()

        accuracy = self._evaluate()
        return total_loss.item(), accuracy

    def _evaluate(self) -> float:
        """Compute current AIA accuracy (deterministic argmax)."""
        with torch.no_grad():
            probs = torch.sigmoid(self.sensitive_logits.detach()).squeeze()
            predicted = (probs > 0.5).float()
            predicted = predicted * (self.s_max - self.s_min) + self.s_min
            true_attr = self.true_features[:, self.sensitive_attr_id]
            accuracy = (torch.round(predicted) == torch.round(true_attr)).float().mean().item()
        return accuracy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_attack(self, num_iterations: int = 10000) -> float:
        """
        Run the gradient-based AIA for ``num_iterations`` steps.

        Args:
            num_iterations: Number of SGD steps on the logits.

        Returns:
            Final AIA accuracy.
        """
        if not self._round_data:
            logger.warning("No round data registered. Call add_round() first.")
            return 0.5

        best_acc = 0.0
        log_every = max(1, num_iterations // 10)

        for it in range(num_iterations):
            loss, acc = self._iteration()
            best_acc = max(best_acc, acc)

            if it % log_every == 0:
                logger.debug(f"  Grad-AIA iter {it}/{num_iterations} | loss={loss:.4f} | acc={acc:.4f}")

        final_acc = self._evaluate()
        logger.info(f"Gradient-based AIA finished | accuracy={final_acc:.4f} (best={best_acc:.4f})")
        return final_acc

    def evaluate_attack(self) -> float:
        return self._evaluate()

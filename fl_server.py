"""
Flower FL Server strategy — FedAvg with Attribute Inference Attack hooks.

Implements a custom Flower Strategy (FedAvgWithAIA) that:
  - Acts as a standard FedAvg server during normal rounds
  - Intercepts client updates to build the adversary's eavesdrop log
  - Optionally injects malicious models during active attack rounds (Algorithm 3)
  - Triggers model-based and gradient-based AIAs at the end of training

The strategy targets ONE specific client (the attacked client) while
allowing all other clients to train normally.
"""

import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Metrics,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from models import get_flat_params, set_flat_params, count_parameters
from attacks.model_based_aia import (
    ModelBasedAIA,
    LinearModelReconstructionAttack,
    ActiveModelReconstructionAttack,
)
from attacks.gradient_based_aia import GradientBasedAIA
from utils import save_checkpoint, save_results, fedavg_aggregate

logger = logging.getLogger(__name__)


class FedAvgWithAIA(fl.server.strategy.Strategy):
    """
    FedAvg strategy augmented with Attribute Inference Attack (AIA) capabilities.

    The server performs standard FedAvg aggregation but also:
      - Stores (global_model, local_model) pairs for the targeted client
        (passive eavesdropping)
      - In active mode, replaces the global model sent to the targeted client
        with an adversarial model after `active_start_round` rounds

    AIA evaluation is triggered at the end of FL training via `run_aia()`.

    Args:
        model_init_fn: Callable returning a fresh nn.Module.
        initial_parameters: Starting model parameters (Flower Parameters object).
        targeted_client_id: ID of the client being attacked.
        attack_mode: 'passive', 'active', or 'gradient'. Default 'passive'.
        active_start_round: Round after which active attack begins.
        active_rounds: Number of active attack rounds. Default 50.
        adam_lr: Adam learning rate for active attack. Default 1.0.
        adam_beta1: Adam β₁. Default 0.9.
        adam_beta2: Adam β₂. Default 0.999.
        checkpoint_dir: Root directory for model checkpoints.
        fraction_fit: Fraction of clients sampled per round.
        fraction_evaluate: Fraction of clients used for evaluation.
        min_fit_clients: Minimum number of clients for training.
        min_evaluate_clients: Minimum clients for evaluation.
        min_available_clients: Minimum available clients before training starts.
        device: Torch device string.
    """

    def __init__(
        self,
        model_init_fn,
        initial_parameters: Parameters,
        targeted_client_id: str,
        attack_mode: str = "passive",
        active_start_round: int = 50,
        active_rounds: int = 50,
        adam_lr: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        checkpoint_dir: str = "checkpoints",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        device: str = "cpu",
    ):
        super().__init__()

        assert attack_mode in ("passive", "active", "gradient"), \
            f"attack_mode must be 'passive', 'active', or 'gradient'"

        self.model_init_fn = model_init_fn
        self.initial_parameters = initial_parameters
        self.targeted_client_id = targeted_client_id
        self.attack_mode = attack_mode
        self.active_start_round = active_start_round
        self.active_rounds = active_rounds
        self.checkpoint_dir = checkpoint_dir
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.device = device

        # Current global model parameters (numpy arrays, Flower convention)
        self._current_params: List[np.ndarray] = parameters_to_ndarrays(initial_parameters)
        self._round = 0

        # Eavesdrop log: {round_id: {"global": path, "local": path}}
        self._global_paths: Dict[int, str] = {}
        self._local_paths: Dict[int, str] = {}

        # Flat parameter lists for LMRA (Algorithm 2)
        self._eavesdropped_global: List[torch.Tensor] = []
        self._eavesdropped_local: List[torch.Tensor] = []

        # Active attack state
        self._active_attacker: Optional[ActiveModelReconstructionAttack] = None
        self._is_active_round = False
        self._active_round_count = 0
        self._adam_lr = adam_lr
        self._adam_beta1 = adam_beta1
        self._adam_beta2 = adam_beta2

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Items 1-4: universal metrics tracked in AIA mode ──────────────
        self._accuracy_history: List[float] = []   # global test accuracy per round
        self._loss_history: List[float] = []        # global test loss per round
        self._convergence_round: Optional[int] = None  # round where loss drops ≤ 95% of initial
        self._total_params: Optional[int] = None   # filled on first aggregate_fit

    # ------------------------------------------------------------------
    # Flower Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure each client's fit instruction, injecting malicious models in active mode."""
        self._round = server_round

        # Sample clients
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        # Determine if this is an active attack round
        self._is_active_round = (
            self.attack_mode == "active"
            and server_round > self.active_start_round
            and self._active_round_count < self.active_rounds
        )

        fit_ins_list = []
        for client in clients:
            if (
                self._is_active_round
                and client.cid == self.targeted_client_id
                and self._active_attacker is not None
            ):
                # Inject adversarial model parameters to the targeted client
                adv_flat = self._active_attacker.get_malicious_model_params()
                adv_model = self.model_init_fn()
                set_flat_params(adv_model, adv_flat)
                adv_params = ndarrays_to_parameters(
                    [p.data.cpu().numpy() for p in adv_model.parameters()]
                )
                ins = FitIns(parameters=adv_params, config={"round": server_round, "active": True})
            else:
                ins = FitIns(parameters=parameters, config={"round": server_round, "active": False})

            fit_ins_list.append((client, ins))

        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates with standard FedAvg.
        Also:
          - Records global/local model pairs for the targeted client (eavesdrop)
          - Updates active attacker if in active attack mode
        """
        if not results:
            return None, {}

        # Save global model checkpoint (before this round's aggregation)
        global_ckpt_path = os.path.join(
            self.checkpoint_dir, "server", f"global_r{server_round}.pt"
        )
        os.makedirs(os.path.dirname(global_ckpt_path), exist_ok=True)
        global_model = self.model_init_fn()
        for p, arr in zip(global_model.parameters(), self._current_params):
            p.data.copy_(torch.from_numpy(arr))
        save_checkpoint(global_model, global_ckpt_path)

        # Process each client result
        weights_results = []
        targeted_local_flat: Optional[torch.Tensor] = None
        global_flat = get_flat_params(global_model)

        for client, fit_res in results:
            local_params_np = parameters_to_ndarrays(fit_res.parameters)
            local_model = self.model_init_fn()
            for p, arr in zip(local_model.parameters(), local_params_np):
                p.data.copy_(torch.from_numpy(arr))
            local_flat = get_flat_params(local_model)

            if client.cid == self.targeted_client_id:
                # --- Eavesdrop: record (global, local) pair ---
                self._eavesdropped_global.append(global_flat.clone())
                self._eavesdropped_local.append(local_flat.clone())
                self._global_paths[server_round] = global_ckpt_path

                # Save local checkpoint
                local_ckpt_path = os.path.join(
                    self.checkpoint_dir, "targeted_client", f"local_r{server_round}.pt"
                )
                os.makedirs(os.path.dirname(local_ckpt_path), exist_ok=True)
                save_checkpoint(local_model, local_ckpt_path)
                self._local_paths[server_round] = local_ckpt_path

                targeted_local_flat = local_flat.clone()

                # --- Active attack update ---
                if self._is_active_round and self._active_attacker is not None:
                    self._active_attacker.update(targeted_local_flat)
                    self._active_round_count += 1
                    logger.debug(
                        f"Active attack round {self._active_round_count}/{self.active_rounds} "
                        f"completed (server round {server_round})"
                    )

                # Initialise active attacker one round BEFORE active rounds begin,
                # so it is ready when configure_fit is called for active_start_round + 1.
                # Also handle the edge case where active_start_round == 1 (round 0 never
                # fires), by allowing initialisation at active_start_round itself.
                if (
                    self.attack_mode == "active"
                    and server_round in (self.active_start_round - 1, self.active_start_round)
                    and self._active_attacker is None
                    and targeted_local_flat is not None
                ):
                    self._active_attacker = ActiveModelReconstructionAttack(
                        initial_model_params=targeted_local_flat,
                        lr=self._adam_lr,
                        beta1=self._adam_beta1,
                        beta2=self._adam_beta2,
                        device=self.device,
                    )
                    logger.info(
                        f"Active attacker initialised at round {server_round} "
                        f"targeting client '{self.targeted_client_id}'"
                    )

            weights_results.append((local_params_np, fit_res.num_examples))

        # FedAvg aggregation
        aggregated = aggregate(weights_results)
        self._current_params = aggregated

        return ndarrays_to_parameters(aggregated), {"round": server_round}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
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
        loss_aggregated = weighted_loss_avg(
            [(r.num_examples, r.loss) for _, r in results]
        )
        # ── Items 1-3: track loss, accuracy, convergence in AIA mode ──────
        self._loss_history.append(float(loss_aggregated))

        # Aggregate accuracy from client metrics (if present)
        total_examples = sum(r.num_examples for _, r in results)
        mse_sum = sum(
            r.metrics.get("mse_loss", 0.0) * r.num_examples for _, r in results
        )
        avg_mse = mse_sum / max(total_examples, 1)
        self._accuracy_history.append(float(avg_mse))

        # Convergence round: first round where MSE drops to ≤95% of the initial MSE.
        # Using a loss-decrease threshold instead of a fixed value is correct for
        # regression with a standardised target (MSE is not bounded in [0,1]).
        if (self._convergence_round is None
                and len(self._loss_history) >= 2
                and self._loss_history[-1] <= 0.95 * self._loss_history[0]):
            self._convergence_round = server_round
            logger.info(f"Convergence (5% loss drop) reached at round {server_round}")

        return loss_aggregated, {"round": server_round}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    # ------------------------------------------------------------------
    # Client sampling helpers (standard FedAvg)
    # ------------------------------------------------------------------

    def num_fit_clients(self, num_available: int) -> Tuple[int, int]:
        num_clients = max(int(num_available * self.fraction_fit), self.min_fit_clients)
        return num_clients, self.min_fit_clients

    def num_evaluation_clients(self, num_available: int) -> Tuple[int, int]:
        num_clients = max(int(num_available * self.fraction_evaluate), self.min_evaluate_clients)
        return num_clients, self.min_evaluate_clients

    # ------------------------------------------------------------------
    # AIA pipeline
    # ------------------------------------------------------------------

    def run_aia(
        self,
        targeted_dataset,
        sensitive_attr_id: int,
        model_type: str = "passive",
        nn_model_type: str = "neural_network",
        num_grad_aia_iterations: int = 5000,
        max_grad_rounds: int = 10,
    ) -> Dict[str, float]:
        """
        Run the full AIA pipeline after FL training completes.

        Steps:
          1. Reconstruct the targeted client's optimal local model.
             - Linear model  → LMRA (Algorithm 2)
             - Neural network → last returned local model (paper Sec. 5.3)
             - Active attack  → Adam-emulated model (Algorithm 3)
          2. Run model-based AIA (Eq. 3) on the reconstructed model.
          3. Run gradient-based AIA baseline (Lyu & Chen 2021) on a
             sub-sampled set of at most `max_grad_rounds` eavesdropped rounds.
          4. Run model-based AIA on the server's final global model.

        Args:
            targeted_dataset:       Dataset of the targeted client (train split).
            sensitive_attr_id:      Index of the sensitive attribute column.
            model_type:             Unused legacy arg; kept for API compatibility.
            nn_model_type:          'neural_network' or 'linear'. Controls which
                                    passive reconstruction strategy is used.
            num_grad_aia_iterations: SGD iterations for gradient-based AIA.
            max_grad_rounds:        Maximum eavesdropped rounds fed to the
                                    gradient-based AIA (avoids O(T) per-iter cost).

        Returns:
            Dict with keys 'ours', 'global_model', 'grad_passive' and their
            AIA accuracies in [0, 1].
        """
        results: Dict[str, float] = {}
        criterion = nn.MSELoss()

        logger.info("=" * 60)
        logger.info("Running Attribute Inference Attacks...")
        logger.info(f"  Targeted client : {self.targeted_client_id}")
        logger.info(f"  Attack mode     : {self.attack_mode}")
        logger.info(f"  nn_model_type   : {nn_model_type}")
        logger.info(f"  Eavesdropped    : {len(self._eavesdropped_global)} rounds")
        logger.info("=" * 60)

        # ----------------------------------------------------------
        # 1. Reconstruct optimal local model
        # ----------------------------------------------------------
        reconstructed_model = self.model_init_fn().to(self.device)

        if self.attack_mode in ("passive", "gradient"):
            if nn_model_type == "linear" and len(self._eavesdropped_global) >= 2:
                # LMRA (Algorithm 2) is only valid for linear least-squares regression.
                # For neural networks this produces garbage weights — DO NOT use here.
                logger.info("Passive reconstruction: running LMRA (Algorithm 2) [linear model]...")
                lmra = LinearModelReconstructionAttack(
                    global_params_list=self._eavesdropped_global,
                    local_params_list=self._eavesdropped_local,
                )
                reconstructed_flat = lmra.reconstruct()
                set_flat_params(reconstructed_model, reconstructed_flat.to(self.device))
                logger.info("LMRA reconstruction complete.")
            elif self._eavesdropped_local:
                # FIX: For neural networks the paper (Sec. 5.3) states:
                # "a passive adversary uses the last-returned model from the targeted client."
                # Previously LMRA was applied unconditionally here, which produced garbage
                # weights for neural networks (underdetermined system with ~2000 unknowns
                # but only ~100 equations), causing NaN losses in ModelBasedAIA and making
                # it always predict s_max — giving accuracy == dominant-class fraction,
                # identical to the global model baseline.
                logger.info(
                    "Passive reconstruction: using last returned local model "
                    "[neural network — paper Sec. 5.3]..."
                )
                set_flat_params(reconstructed_model, self._eavesdropped_local[-1].to(self.device))
            else:
                logger.warning("No eavesdropped local models available — falling back to global model.")
                for p, arr in zip(reconstructed_model.parameters(), self._current_params):
                    p.data.copy_(torch.from_numpy(arr).to(self.device))

        elif self.attack_mode == "active" and self._active_attacker is not None:
            logger.info("Active reconstruction: using Adam-emulated local model (Algorithm 3)...")
            reconstructed_flat = self._active_attacker.reconstructed_params
            set_flat_params(reconstructed_model, reconstructed_flat.to(self.device))
            logger.info("Active reconstruction complete.")

        else:
            # Fallback: use last observed local model
            if self._eavesdropped_local:
                set_flat_params(reconstructed_model, self._eavesdropped_local[-1].to(self.device))
                logger.warning("Using last observed local model as reconstruction fallback.")
            else:
                logger.warning("No eavesdropped data available — using global model.")
                for p, arr in zip(reconstructed_model.parameters(), self._current_params):
                    p.data.copy_(torch.from_numpy(arr).to(self.device))

        # ----------------------------------------------------------
        # 2. Model-based AIA on reconstructed local model (our attack)
        # ----------------------------------------------------------
        logger.info("Running model-based AIA on reconstructed local model...")
        aia = ModelBasedAIA(
            model=reconstructed_model,
            dataset=targeted_dataset,
            sensitive_attr_id=sensitive_attr_id,
            criterion=criterion,
            device=self.device,
        )
        aia.execute_attack()
        our_acc = aia.evaluate_attack()
        our_mse = aia.evaluate_attack_mse()  # Item 5: reconstruction MSE
        results["ours"] = our_acc
        results["ours_mse"] = our_mse
        logger.info(f"  Our AIA accuracy: {our_acc:.4f} ({our_acc * 100:.2f}%)")
        logger.info(f"  Our AIA MSE:      {our_mse:.6f}")

        # ----------------------------------------------------------
        # 3. Model-based AIA on the global model (lower bound)
        # ----------------------------------------------------------
        logger.info("Running model-based AIA on global model (baseline)...")
        global_model = self.model_init_fn().to(self.device)
        for p, arr in zip(global_model.parameters(), self._current_params):
            p.data.copy_(torch.from_numpy(arr).to(self.device))

        aia_global = ModelBasedAIA(
            model=global_model,
            dataset=targeted_dataset,
            sensitive_attr_id=sensitive_attr_id,
            criterion=criterion,
            device=self.device,
        )
        aia_global.execute_attack()
        global_acc = aia_global.evaluate_attack()
        global_mse = aia_global.evaluate_attack_mse()  # Item 5: global model MSE
        results["global_model"] = global_acc
        results["global_model_mse"] = global_mse
        logger.info(f"  Global model AIA accuracy: {global_acc:.4f} ({global_acc * 100:.2f}%)")
        logger.info(f"  Global model AIA MSE:      {global_mse:.6f}")

        # ----------------------------------------------------------
        # 4. Gradient-based AIA baseline (Lyu & Chen 2021)
        # ----------------------------------------------------------
        if len(self._eavesdropped_global) >= 1:
            logger.info("Running gradient-based AIA baseline...")

            # FIX: Previously ALL eavesdropped rounds (e.g. 100) were added to
            # GradientBasedAIA._round_data.  Every optimisation iteration then
            # instantiated and forward/backward-passed through 100 models, making
            # the total cost O(T × iterations) ≈ 100 × 5000 = 500 000 model passes.
            # This caused the run to hang or OOM, so 'grad_passive' was never written
            # to results and the table showed N/A for all Grad cells.
            #
            # Fix: evenly sub-sample at most `max_grad_rounds` rounds so the per-
            # iteration cost stays bounded regardless of training length.
            n_avail = len(self._eavesdropped_global)
            if n_avail <= max_grad_rounds:
                selected_indices = list(range(n_avail))
            else:
                step = n_avail / max_grad_rounds
                selected_indices = [int(i * step) for i in range(max_grad_rounds)]

            logger.info(
                f"  Using {len(selected_indices)}/{n_avail} eavesdropped rounds "
                f"for gradient-based AIA (max_grad_rounds={max_grad_rounds})."
            )

            grad_aia = GradientBasedAIA(
                model_init_fn=self.model_init_fn,
                dataset=targeted_dataset,
                sensitive_attr_id=sensitive_attr_id,
                criterion=criterion,
                device=self.device,
                learning_rate=1e4,
                gumbel_tau=1.0,
            )
            for idx in selected_indices:
                g_flat = self._eavesdropped_global[idx]
                l_flat = self._eavesdropped_local[idx]
                g_model = self.model_init_fn()
                set_flat_params(g_model, g_flat)
                l_model = self.model_init_fn()
                set_flat_params(l_model, l_flat)
                grad_aia.add_round(
                    global_params=[p.data.clone() for p in g_model.parameters()],
                    local_params=[p.data.clone() for p in l_model.parameters()],
                )

            grad_acc = grad_aia.execute_attack(num_iterations=num_grad_aia_iterations)
            results["grad_passive"] = grad_acc
            logger.info(f"  Gradient-based AIA accuracy: {grad_acc:.4f} ({grad_acc * 100:.2f}%)")

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        logger.info("=" * 60)
        logger.info("AIA Results Summary:")
        for name, acc in results.items():
            logger.info(f"  {name:25s}: {acc * 100:.2f}%" if "mse" not in name
                        else f"  {name:25s}: {acc:.6f}")
        logger.info("=" * 60)

        # ── Items 1-4: attach universal metrics to results ─────────────────
        # Item 1: global test accuracy history
        results["accuracy_history"] = self._accuracy_history
        results["final_accuracy"] = self._accuracy_history[-1] if self._accuracy_history else None

        # Item 2: global test loss history
        results["loss_history"] = self._loss_history
        results["final_loss"] = self._loss_history[-1] if self._loss_history else None

        # Item 3: convergence round (first round where MSE ≤ 95% of initial MSE)
        results["convergence_round"] = self._convergence_round

        # Item 4: communication cost in MB
        # Each round: n_clients × 2 (send+receive) × model_size_bytes
        if self._total_params is None:
            tmp = self.model_init_fn()
            self._total_params = count_parameters(tmp)
        # float32 = 4 bytes per param; multiply by rounds × clients × 2 directions
        n_rounds = self._round
        n_clients_est = self.min_fit_clients  # actual number of participating clients
        model_bytes = self._total_params * 4  # float32
        comm_cost_mb = (n_rounds * n_clients_est * 2 * model_bytes) / (1024 ** 2)
        results["communication_cost_mb"] = round(comm_cost_mb, 4)
        logger.info(f"  Communication cost (est.): {comm_cost_mb:.4f} MB")

        return results
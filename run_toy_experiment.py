"""
run_toy_experiment.py — Toy linear regression AIA experiment.

Replicates Figure 2 and Appendix B.3 of the paper:
  "Attribute Inference Attacks for Federated Regression Tasks"
  (Diana et al., AAAI 2025)

Setup (matching paper Appendix B.3):
  - 2 clients, each with 1024 training samples
  - d = 11 parameters (10 features + 1 bias in LinearModel)
  - 9 numerical features ~ Uniform[0, 1), 1 binary sensitive feature ~ {-1,+1}
  - FedAvg, 1 local epoch, 300 communication rounds
  - Passive adversary eavesdrops d+1 = 12 messages (from rounds spaced evenly)
  - Batch sizes tested: 64, 256, 1024  (paper: Figure 2 x-axis)
  - 5 random seeds per batch size

Attack pipeline (Algorithm 2 + Eq. 3):
  1. LMRA (LinearModelReconstructionAttack, Algorithm 2):
     Reconstruct client's optimal local linear model from eavesdropped messages.
  2. Model-based AIA (ModelBasedAIA, Eq. 3):
     Infer the binary sensitive attribute by checking which value minimises loss.

Metrics reported:
  - ||θ̂* − θ*||₂  (reconstruction error, Figure 2 left)
  - AIA accuracy  (fraction of correct sensitive attribute inferences, Figure 2 right)

Outputs:
  - results/toy_experiment_results.json   (full per-seed + aggregated results)
  - results/toy_experiment_results.csv    (flat CSV, one row per batch_size)
  - results/figures/toy_figure2.png       (Figure 2 reproduction: recon error + AIA acc)

Usage:
  python run_toy_experiment.py [options]

  # Quick single run (paper defaults)
  python run_toy_experiment.py

  # Vary batch sizes and seeds as in Figure 2
  python run_toy_experiment.py --batch_sizes 64 256 1024 --n_seeds 5

  # Reproduce Figure 2 exactly
  python run_toy_experiment.py --reproduce_figure2

  # Custom settings
  python run_toy_experiment.py --batch_sizes 32 128 --n_rounds 300 --n_seeds 3
"""

import argparse
import csv
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure the flower project root is on sys.path so that sibling packages
# (datasets/, attacks/, models.py, utils.py) are importable regardless of
# the working directory from which this script is invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from datasets.toy import FederatedToyDataset
from models import LinearModel, get_flat_params, set_flat_params
from attacks.model_based_aia import LinearModelReconstructionAttack, ModelBasedAIA
from utils import set_seed, configure_logging, save_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults matching paper Appendix B.3
# ---------------------------------------------------------------------------
PAPER_N_CLIENTS = 2
PAPER_N_TRAIN = 1024
PAPER_N_ROUNDS = 300
PAPER_LOCAL_EPOCHS = 1
PAPER_BATCH_SIZES = [64, 256, 1024]
PAPER_N_SEEDS = 5
PAPER_LR = 5e-3          # Appendix C.1: lr=5e-3 for least-squares
PAPER_N_FEATURES = 10    # 9 numerical + 1 binary → LinearModel input_dim=10, d=11 params
PAPER_EAVESDROP_ROUNDS = 12   # d + 1 = 11 + 1  (paper: "eavesdropped d+1 messages")
PAPER_EAVESDROP_SPACING = 20  # paper: T = {i * 20 | i ∈ {0,...,11}}


# ---------------------------------------------------------------------------
# FedAvg simulation (pure PyTorch, no Flower overhead for this small experiment)
# ---------------------------------------------------------------------------

def fedavg_local_update(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    local_epochs: int,
    device: str,
) -> nn.Module:
    """Run FedAvg local update (Algorithm 4) and return updated model."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    return model


def run_fedavg(
    fed_dataset: FederatedToyDataset,
    batch_size: int,
    n_rounds: int,
    local_epochs: int,
    lr: float,
    device: str,
    eavesdrop_rounds: Optional[List[int]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], nn.Module]:
    """
    Simulate FedAvg on a toy federated dataset.

    Returns:
        global_params_history: flat global param tensors at each eavesdropped round
        local_params_history:  flat local  param tensors at each eavesdropped round
        final_global_model:    the global model after n_rounds
    """
    input_dim = fed_dataset.input_dim
    n_clients = fed_dataset.num_clients()

    # Initialise global model
    global_model = LinearModel(input_dimension=input_dim).to(device)
    nn.init.zeros_(global_model.linear.weight)
    nn.init.zeros_(global_model.linear.bias)

    global_params_history: List[torch.Tensor] = []
    local_params_history: List[torch.Tensor] = []

    if eavesdrop_rounds is None:
        eavesdrop_rounds = list(range(n_rounds))

    eavesdrop_set = set(eavesdrop_rounds)

    # Use client 0 as the attacked/targeted client (consistent with paper)
    attacked_client = 0

    for t in range(n_rounds):
        # Server broadcasts global model to all clients
        global_flat = get_flat_params(global_model).clone()

        # Each client performs local update
        client_params: List[Tuple[List[np.ndarray], int]] = []
        attacked_local_flat: Optional[torch.Tensor] = None

        for cid in range(n_clients):
            local_model = deepcopy(global_model)
            loader = fed_dataset.get_dataloader(cid, mode="train", batch_size=batch_size, shuffle=True)
            local_model = fedavg_local_update(local_model, loader, lr, local_epochs, device)
            local_flat = get_flat_params(local_model).clone()

            if cid == attacked_client:
                # Passive adversary eavesdrops on attacked client's messages
                if t in eavesdrop_set:
                    global_params_history.append(global_flat.cpu())
                    local_params_history.append(local_flat.cpu())
                attacked_local_flat = local_flat

            n_samples = len(fed_dataset.get_dataset(cid, "train"))
            client_params.append(([p.data.cpu().numpy() for p in local_model.parameters()], n_samples))

        # FedAvg aggregation (weighted average by number of samples)
        total = sum(n for _, n in client_params)
        agg_params = [
            np.sum([p[i] * n / total for p, n in client_params], axis=0)
            for i in range(len(client_params[0][0]))
        ]
        for param, agg in zip(global_model.parameters(), agg_params):
            param.data.copy_(torch.from_numpy(agg).to(device))

        if (t + 1) % 50 == 0:
            logger.debug(f"  Round {t+1}/{n_rounds} done")

    return global_params_history, local_params_history, global_model


# ---------------------------------------------------------------------------
# Single experiment: one (batch_size, seed) pair
# ---------------------------------------------------------------------------

def run_single(
    batch_size: int,
    seed: int,
    n_rounds: int = PAPER_N_ROUNDS,
    local_epochs: int = PAPER_LOCAL_EPOCHS,
    lr: float = PAPER_LR,
    n_clients: int = PAPER_N_CLIENTS,
    n_train: int = PAPER_N_TRAIN,
    n_features: int = PAPER_N_FEATURES,
    device: str = "cpu",
) -> Dict:
    """
    Run one full experiment: FL simulation → LMRA → model-based AIA.

    Returns dict with:
      - 'reconstruction_error': ||θ̂* − θ*||₂
      - 'aia_accuracy':         fraction of correctly inferred sensitive attrs
      - 'global_aia_accuracy':  AIA accuracy on the final global model
    """
    set_seed(seed)

    # ---- Dataset ----
    fed_dataset = FederatedToyDataset(
        n_clients=n_clients,
        n_train_samples=n_train,
        n_test_samples=128,
        n_numerical_features=n_features - 1,   # 9 numerical, 1 binary
        n_binary_features=1,
        noise_std=0.1,
        seed=seed,
    )
    sensitive_attr_id = fed_dataset.sensitive_attr_id
    input_dim = fed_dataset.input_dim

    # ---- Determine eavesdrop rounds (paper: T = {i*20 | i ∈ {0,...,11}}) ----
    d_params = input_dim + 1  # LinearModel: input_dim weights + 1 bias = d
    n_eavesdrop = d_params + 1  # d+1 messages needed (Proposition 2)
    eavesdrop_rounds = [i * PAPER_EAVESDROP_SPACING for i in range(n_eavesdrop)]
    # Clip to valid range
    eavesdrop_rounds = [r for r in eavesdrop_rounds if r < n_rounds]

    logger.debug(
        f"B={batch_size}, seed={seed} | d_params={d_params} | "
        f"eavesdrop_rounds={eavesdrop_rounds}"
    )

    # ---- FedAvg simulation ----
    global_params_hist, local_params_hist, final_global = run_fedavg(
        fed_dataset=fed_dataset,
        batch_size=batch_size,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        lr=lr,
        device=device,
        eavesdrop_rounds=eavesdrop_rounds,
    )

    # ---- Compute empirical optimal local model (θ* approximation) ----
    opt_model = LinearModel(input_dimension=input_dim).to(device)
    opt_optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-2)
    opt_criterion = nn.MSELoss()
    full_loader = fed_dataset.get_dataloader(0, mode="train", batch_size=n_train, shuffle=False)
    for _ in range(2000):
        for x, y in full_loader:
            x, y = x.to(device).float(), y.to(device).float()
            opt_optimizer.zero_grad()
            opt_criterion(opt_model(x).squeeze(), y).backward()
            opt_optimizer.step()
    theta_star = get_flat_params(opt_model).cpu()

    # ---- LMRA (Algorithm 2): reconstruct optimal local model ----
    if len(global_params_hist) < 2:
        logger.warning(
            f"Not enough eavesdropped rounds ({len(global_params_hist)}) for LMRA. "
            "Using last local model instead."
        )
        reconstructed_flat = local_params_hist[-1] if local_params_hist else theta_star
    else:
        lmra = LinearModelReconstructionAttack(
            global_params_list=global_params_hist,
            local_params_list=local_params_hist,
        )
        reconstructed_flat = lmra.reconstruct().cpu()

    reconstruction_error = torch.norm(reconstructed_flat - theta_star).item()
    logger.debug(f"  Reconstruction error ||θ̂* − θ*||₂ = {reconstruction_error:.4f}")

    # ---- Model-based AIA (Eq. 3) on reconstructed model ----
    reconstructed_model = LinearModel(input_dimension=input_dim).to(device)
    set_flat_params(reconstructed_model, reconstructed_flat.to(device))

    train_dataset = fed_dataset.get_dataset(0, mode="train")
    aia = ModelBasedAIA(
        model=reconstructed_model,
        dataset=train_dataset,
        sensitive_attr_id=sensitive_attr_id,
        criterion=nn.MSELoss(),
        device=device,
    )
    aia.execute_attack()
    aia_acc = aia.evaluate_attack()
    logger.debug(f"  AIA accuracy (our, reconstructed model) = {aia_acc:.4f}")

    # ---- Model-based AIA on final global model (baseline) ----
    aia_global = ModelBasedAIA(
        model=final_global,
        dataset=train_dataset,
        sensitive_attr_id=sensitive_attr_id,
        criterion=nn.MSELoss(),
        device=device,
    )
    aia_global.execute_attack()
    global_aia_acc = aia_global.evaluate_attack()
    logger.debug(f"  AIA accuracy (global model) = {global_aia_acc:.4f}")

    return {
        "reconstruction_error": reconstruction_error,
        "aia_accuracy": aia_acc,
        "global_aia_accuracy": global_aia_acc,
        "batch_size": batch_size,
        "seed": seed,
        "n_eavesdropped": len(global_params_hist),
    }


# ---------------------------------------------------------------------------
# Full sweep over batch sizes × seeds
# ---------------------------------------------------------------------------

def run_sweep(
    batch_sizes: List[int],
    n_seeds: int,
    n_rounds: int = PAPER_N_ROUNDS,
    local_epochs: int = PAPER_LOCAL_EPOCHS,
    lr: float = PAPER_LR,
    device: str = "cpu",
    results_dir: str = "results",
) -> Dict:
    """
    Run the full paper sweep (Figure 2): batch_sizes × n_seeds.

    Saves:
      - results/toy_experiment_results.json
      - results/toy_experiment_results.csv
      - results/figures/toy_figure2.png

    Returns aggregated results dict with mean ± std per batch size.
    """
    all_results: Dict[int, List[Dict]] = {b: [] for b in batch_sizes}

    for batch_size in batch_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch size B = {batch_size}")
        logger.info(f"{'='*60}")

        for seed_idx in range(n_seeds):
            seed = seed_idx  # paper uses seeds 0..4
            logger.info(f"  Seed {seed_idx+1}/{n_seeds} ...")
            result = run_single(
                batch_size=batch_size,
                seed=seed,
                n_rounds=n_rounds,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
            )
            all_results[batch_size].append(result)
            logger.info(
                f"    recon_err={result['reconstruction_error']:.4f} | "
                f"aia_acc={result['aia_accuracy']:.4f} | "
                f"global_aia_acc={result['global_aia_accuracy']:.4f}"
            )

    # Aggregate
    summary: Dict = {"per_seed": {}, "aggregated": {}}

    for batch_size, runs in all_results.items():
        key = f"B={batch_size}"
        summary["per_seed"][key] = runs

        recon_errs = [r["reconstruction_error"] for r in runs]
        aia_accs = [r["aia_accuracy"] for r in runs]
        global_accs = [r["global_aia_accuracy"] for r in runs]

        summary["aggregated"][key] = {
            "reconstruction_error_mean": float(np.mean(recon_errs)),
            "reconstruction_error_std":  float(np.std(recon_errs)),
            "aia_accuracy_mean":         float(np.mean(aia_accs)),
            "aia_accuracy_std":          float(np.std(aia_accs)),
            "global_aia_accuracy_mean":  float(np.mean(global_accs)),
            "global_aia_accuracy_std":   float(np.std(global_accs)),
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Toy Linear Regression AIA Experiment — Results (Figure 2 equivalent)")
    print(f"{'='*70}")
    print(f"{'Batch Size':>12} | {'Recon Err (mean±std)':>22} | "
          f"{'AIA Acc (mean±std)':>22} | {'Global AIA':>12}")
    print("-" * 70)
    for batch_size in batch_sizes:
        key = f"B={batch_size}"
        agg = summary["aggregated"][key]
        print(
            f"{batch_size:>12} | "
            f"{agg['reconstruction_error_mean']:>8.4f} ± {agg['reconstruction_error_std']:<8.4f} | "
            f"{agg['aia_accuracy_mean']*100:>8.2f}% ± {agg['aia_accuracy_std']*100:<8.2f}% | "
            f"{agg['global_aia_accuracy_mean']*100:>8.2f}%"
        )
    print(f"{'='*70}\n")

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(results_dir, "toy_experiment_results.json")
    save_results(summary, json_path)
    logger.info(f"JSON results saved to {json_path}")

    # ------------------------------------------------------------------
    # Save CSV  (one row per batch_size with mean ± std columns)
    # ------------------------------------------------------------------
    csv_path = os.path.join(results_dir, "toy_experiment_results.csv")
    csv_rows = []
    for batch_size in batch_sizes:
        key = f"B={batch_size}"
        agg = summary["aggregated"][key]
        csv_rows.append({
            "batch_size":                    batch_size,
            "n_seeds":                       n_seeds,
            "n_rounds":                      n_rounds,
            "recon_error_mean":              round(agg["reconstruction_error_mean"], 6),
            "recon_error_std":               round(agg["reconstruction_error_std"],  6),
            "aia_accuracy_mean_pct":         round(agg["aia_accuracy_mean"] * 100, 2),
            "aia_accuracy_std_pct":          round(agg["aia_accuracy_std"]  * 100, 2),
            "global_aia_accuracy_mean_pct":  round(agg["global_aia_accuracy_mean"] * 100, 2),
            "global_aia_accuracy_std_pct":   round(agg["global_aia_accuracy_std"]  * 100, 2),
        })

    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"CSV results saved to {csv_path}")

    # ------------------------------------------------------------------
    # Save Figure (Figure 2 reproduction: left = recon error, right = AIA acc)
    # ------------------------------------------------------------------
    _save_figure2(summary, batch_sizes, results_dir)

    return summary


def _save_figure2(summary: Dict, batch_sizes: List[int], results_dir: str) -> None:
    """Generate and save Figure 2 (reconstruction error + AIA accuracy vs batch size)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping figure generation.")
        return

    recon_means, recon_stds = [], []
    aia_means, aia_stds = [], []
    global_means = []

    for b in batch_sizes:
        key = f"B={b}"
        agg = summary["aggregated"][key]
        recon_means.append(agg["reconstruction_error_mean"])
        recon_stds.append(agg["reconstruction_error_std"])
        aia_means.append(agg["aia_accuracy_mean"] * 100)
        aia_stds.append(agg["aia_accuracy_std"] * 100)
        global_means.append(agg["global_aia_accuracy_mean"] * 100)

    x = np.arange(len(batch_sizes))
    labels = [f"B={b}" for b in batch_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Reconstruction Error ---
    ax1.bar(x, recon_means, yerr=recon_stds, capsize=6,
            color="#5B9BD5", edgecolor="white", width=0.5, label="LMRA (Alg. 2)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Batch Size B")
    ax1.set_ylabel(r"$\|\hat{\theta}^* - \theta^*\|_2$")
    ax1.set_title("Model Reconstruction Error")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    # --- Right: AIA Accuracy ---
    ax2.bar(x - 0.2, aia_means, yerr=aia_stds, capsize=6,
            color="#ED7D31", edgecolor="white", width=0.35, label="Our AIA (Reconstructed)")
    ax2.bar(x + 0.2, global_means, capsize=6,
            color="#70AD47", edgecolor="white", width=0.35, label="Global Model AIA")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Batch Size B")
    ax2.set_ylabel("AIA Accuracy (%)")
    ax2.set_title("Attribute Inference Attack Accuracy")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle(
        "Toy Linear Regression AIA Experiment (Paper Figure 2 Reproduction)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "toy_figure2.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy linear regression AIA experiment (paper Figure 2 / Appendix B.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--batch_sizes", type=int, nargs="+", default=PAPER_BATCH_SIZES,
        help="Batch sizes to sweep over (paper: 64 256 1024)."
    )
    p.add_argument(
        "--n_seeds", type=int, default=PAPER_N_SEEDS,
        help="Number of random seeds per batch size (paper: 5)."
    )
    p.add_argument(
        "--n_rounds", type=int, default=PAPER_N_ROUNDS,
        help="Number of FL communication rounds (paper: 300)."
    )
    p.add_argument(
        "--local_epochs", type=int, default=PAPER_LOCAL_EPOCHS,
        help="Local epochs per round (paper: 1)."
    )
    p.add_argument(
        "--lr", type=float, default=PAPER_LR,
        help="SGD learning rate (paper Appendix C.1: 5e-3 for least-squares)."
    )
    p.add_argument(
        "--n_clients", type=int, default=PAPER_N_CLIENTS,
        help="Number of FL clients (paper: 2)."
    )
    p.add_argument(
        "--n_train", type=int, default=PAPER_N_TRAIN,
        help="Training samples per client (paper: 1024)."
    )
    p.add_argument(
        "--n_features", type=int, default=PAPER_N_FEATURES,
        help="Number of raw features per sample (paper: 10 → d=11 params with bias)."
    )
    p.add_argument(
        "--reproduce_figure2", action="store_true",
        help="Run the exact paper Figure 2 settings (B∈{64,256,1024}, 5 seeds)."
    )
    p.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to save results JSON, CSV and figure."
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device."
    )
    p.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.reproduce_figure2:
        logger.info("Reproducing paper Figure 2 (Appendix B.3 settings)...")
        batch_sizes = PAPER_BATCH_SIZES
        n_seeds = PAPER_N_SEEDS
    else:
        batch_sizes = args.batch_sizes
        n_seeds = args.n_seeds

    run_sweep(
        batch_sizes=batch_sizes,
        n_seeds=n_seeds,
        n_rounds=args.n_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        device=args.device,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
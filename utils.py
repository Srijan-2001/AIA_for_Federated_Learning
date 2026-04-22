"""
Shared utility functions for the AIA-FL Flower implementation.
"""

import os
import json
import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
# Model parameter helpers
# ---------------------------------------------------------------------------

def params_to_numpy(params: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert list of torch tensors to numpy arrays (for Flower ndarrays)."""
    return [p.cpu().numpy() for p in params]


def numpy_to_params(ndarrays: List[np.ndarray]) -> List[torch.Tensor]:
    """Convert list of numpy arrays to torch tensors."""
    return [torch.from_numpy(arr) for arr in ndarrays]


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view(p.shape))
        offset += n


def clone_model_params(model: nn.Module) -> List[torch.Tensor]:
    return [p.data.clone() for p in model.parameters()]


def load_params_into_model(model: nn.Module, params: List[torch.Tensor]) -> None:
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Accuracy for binary attributes (round to 0/1)."""
    assert y_pred.shape == y_true.shape
    return (torch.round(y_pred) == y_true).float().mean().item()


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(y_pred, y_true).item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.nn.functional.l1_loss(y_pred, y_true).item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Messages metadata
# ---------------------------------------------------------------------------

def build_messages_metadata(
    global_paths: Dict[int, str],
    local_paths: Dict[int, str],
) -> Dict[str, Dict[str, str]]:
    """
    Build the messages metadata dict expected by attack classes.

    Structure:
        {
            "global": {round_id: path, ...},
            "local":  {round_id: path, ...},
        }
    """
    return {
        "global": {str(k): v for k, v in global_paths.items()},
        "local":  {str(k): v for k, v in local_paths.items()},
    }


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params: List[Tuple[List[torch.Tensor], int]],
) -> List[torch.Tensor]:
    """
    Aggregate model parameters using weighted FedAvg.

    Args:
        client_params: List of (params, num_samples) tuples.

    Returns:
        Aggregated parameter list.
    """
    total_samples = sum(n for _, n in client_params)
    aggregated = [torch.zeros_like(p) for p in client_params[0][0]]

    for params, n in client_params:
        weight = n / total_samples
        for agg, p in zip(aggregated, params):
            agg.add_(p * weight)

    return aggregated


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CSV saving — Item 8: save metrics as .csv
# ---------------------------------------------------------------------------

def save_results_csv(rows: List[Dict], path: str) -> None:
    """
    Save a list of result dicts as a CSV file.

    Args:
        rows: List of dicts, each representing one experiment row.
        path: Destination .csv path.
    """
    import csv
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"CSV results saved to {path}")


def save_multi_seed_csv(all_seed_results: List[Dict], path: str) -> None:
    """
    Given results from multiple seeds, compute mean ± std and save as CSV.

    Args:
        all_seed_results: List of result dicts, one per seed run.
        path: Destination .csv path.
    """
    import csv

    if not all_seed_results:
        return

    # Collect scalar metric keys (skip lists and non-numeric values)
    scalar_keys = [
        k for k, v in all_seed_results[0].items()
        if isinstance(v, (int, float)) and v is not None
    ]

    rows = []
    for key in scalar_keys:
        vals = []
        for r in all_seed_results:
            v = r.get(key)
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        if vals:
            rows.append({
                "metric": key,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n_seeds": len(vals),
                "values": str(vals),
            })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fieldnames = ["metric", "mean", "std", "n_seeds", "values"]
    with open(path, "w", newline="") as f:
        import csv as csv_mod
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Multi-seed CSV saved to {path}")


# ---------------------------------------------------------------------------
# Plotting — Items 9-10: generate PNG/PDF figures
# ---------------------------------------------------------------------------

def plot_accuracy_loss_curves(
    accuracy_history: List[float],
    loss_history: List[float],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot global test accuracy and loss curves over FL rounds.

    Args:
        accuracy_history: Accuracy per round.
        loss_history:     Loss per round.
        title:            Plot title.
        save_path:        Path to save figure (.png or .pdf). If None, shows interactively.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed — skipping plot generation.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    rounds = list(range(1, len(accuracy_history) + 1))
    if accuracy_history:
        ax1.plot(rounds, [a * 100 for a in accuracy_history], marker="o", markersize=2,
                 linewidth=1.5, color="steelblue")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Global Test Accuracy")
        ax1.grid(True, alpha=0.3)

    rounds2 = list(range(1, len(loss_history) + 1))
    if loss_history:
        ax2.plot(rounds2, loss_history, marker="o", markersize=2,
                 linewidth=1.5, color="tomato")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("MSE Loss")
        ax2.set_title("Global Test Loss")
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Accuracy/loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_aia_comparison(
    results_by_method: Dict[str, float],
    dataset_name: str = "",
    attack_mode: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Item 10: Side-by-side bar plot comparing AIA accuracy across methods.

    Shows Grad (baseline), Ours, and Global Model bars.

    Args:
        results_by_method: Dict mapping method name → accuracy (0–1 scale).
        dataset_name:      Dataset label for the plot title.
        attack_mode:       Attack mode label (passive / active).
        save_path:         Path to save figure. If None, shows interactively.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed — skipping AIA comparison plot.")
        return

    # Define display order and colours
    ordered_keys = ["grad_passive", "ours", "global_model"]
    labels_map = {
        "grad_passive": "Grad (baseline)",
        "ours": "Ours",
        "global_model": "Global Model",
    }
    colours = {
        "grad_passive": "#5B9BD5",
        "ours": "#ED7D31",
        "global_model": "#70AD47",
    }

    keys   = [k for k in ordered_keys if k in results_by_method]
    labels = [labels_map.get(k, k) for k in keys]
    values = [results_by_method[k] * 100 for k in keys]
    cols   = [colours.get(k, "#888888") for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=cols, width=0.45, edgecolor="white")

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=9,
        )

    title_parts = [p for p in [dataset_name, attack_mode] if p]
    ax.set_title("AIA Accuracy — " + " | ".join(title_parts) if title_parts else "AIA Accuracy")
    ax.set_ylabel("AIA Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"AIA comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

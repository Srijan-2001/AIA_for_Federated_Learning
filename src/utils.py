"""
utils.py — Shared utilities: seeds, metrics, checkpoints, plotting, results saving.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seeds (Section 2 of instructions)
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds per instruction requirement."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(model: nn.Module, loader, device: str = "cpu") -> Tuple[float, float]:
    """Compute cross-entropy loss and classification accuracy on a DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y.long()).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def communication_cost_mb(model: nn.Module, num_clients: int, num_rounds: int) -> float:
    """Compute total communication cost: model_size_MB × clients × rounds × 2 (up+down)."""
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return size_mb * num_clients * num_rounds * 2


def convergence_round(
    accuracy_history: List[Tuple[int, float]],
    threshold: float = 0.80,
) -> Optional[int]:
    """Return first round where accuracy >= threshold, or None."""
    for rnd, acc in accuracy_history:
        if acc >= threshold:
            return rnd
    return None


def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return (torch.round(y_pred) == y_true).float().mean().item()


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(y_pred, y_true).item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str, extra: Optional[Dict] = None) -> None:
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
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params: List[Tuple[List[torch.Tensor], int]],
) -> List[torch.Tensor]:
    total_samples = sum(n for _, n in client_params)
    aggregated = [torch.zeros_like(p) for p in client_params[0][0]]
    for params, n in client_params:
        w = n / total_samples
        for agg, p in zip(aggregated, params):
            agg.add_(p * w)
    return aggregated


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Plotting (mandatory figures per Section 6.1)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_rounds(
    accuracy_history: List[Tuple[int, float]],
    title: str = "Global Accuracy vs Communication Rounds",
    save_path: Optional[str] = None,
    extra_series: Optional[Dict[str, List[Tuple[int, float]]]] = None,
) -> None:
    """
    Plot global accuracy vs rounds (mandatory figure, Section 6.1).

    Args:
        accuracy_history: List of (round, accuracy) tuples for primary series.
        title: Plot title.
        save_path: If given, save the figure as PNG/PDF (300 DPI minimum).
        extra_series: Additional named series to overlay on the same plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    rounds, accs = zip(*accuracy_history) if accuracy_history else ([], [])
    ax.plot(rounds, [a * 100 for a in accs], label="FedAvg", linewidth=2)

    if extra_series:
        for label, series in extra_series.items():
            if series:
                r, a = zip(*series)
                ax.plot(r, [v * 100 for v in a], label=label, linewidth=2)

    ax.set_xlabel("Communication Rounds", fontsize=12)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    plt.close()


def plot_loss_vs_rounds(
    loss_history: List[Tuple[int, float]],
    title: str = "Global Loss vs Communication Rounds",
    save_path: Optional[str] = None,
) -> None:
    """Plot global loss vs rounds (mandatory figure)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    rounds, losses = zip(*loss_history) if loss_history else ([], [])
    ax.plot(rounds, losses, color="crimson", linewidth=2)
    ax.set_xlabel("Communication Rounds", fontsize=12)
    ax.set_ylabel("Global Test Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    plt.close()


def plot_aia_accuracy_vs_rounds(
    aia_history: List[Tuple[int, float]],
    title: str = "AIA Attack Success Rate vs Rounds",
    save_path: Optional[str] = None,
) -> None:
    """Plot category-specific metric (AIA accuracy) vs rounds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    rounds, accs = zip(*aia_history) if aia_history else ([], [])
    ax.plot(rounds, [a * 100 for a in accs], color="darkorange", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Communication Rounds", fontsize=12)
    ax.set_ylabel("AIA Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_iid_vs_noniid(
    iid_history: List[Tuple[int, float]],
    noniid_history: List[Tuple[int, float]],
    title: str = "IID vs Non-IID Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Mandatory IID vs Non-IID comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if iid_history:
        r, a = zip(*iid_history)
        ax.plot(r, [v * 100 for v in a], label="IID", linewidth=2)
    if noniid_history:
        r, a = zip(*noniid_history)
        ax.plot(r, [v * 100 for v in a], label="Non-IID (Dirichlet)", linewidth=2, linestyle="--")

    ax.set_xlabel("Communication Rounds", fontsize=12)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

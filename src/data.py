"""
data.py — Dataset loading and federated partitioning.

Supports:
  - MNIST, FMNIST (Fashion-MNIST), CIFAR-10, CIFAR-100 (via torchvision)
  - Medical Cost and ACS Income (regression, for AIA experiments)

Partitioning:
  - IID: uniform random split
  - Non-IID Dirichlet: alpha ∈ {0.01, 0.1, 0.5, 1.0}

Seeds: random.seed(42), numpy.seed(42), torch.manual_seed(42) — fixed globally.
"""

import os
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
SEED = 42

def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Torchvision dataset loaders
# ---------------------------------------------------------------------------

_DATASET_REGISTRY = {
    "mnist":    (torchvision.datasets.MNIST,        1,  10),
    "fmnist":   (torchvision.datasets.FashionMNIST, 1,  10),
    "cifar10":  (torchvision.datasets.CIFAR10,      3,  10),
    "cifar100": (torchvision.datasets.CIFAR100,     3, 100),
}

def _get_transforms(dataset_name: str):
    name = dataset_name.lower()
    if name in ("mnist", "fmnist"):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif name == "cifar10":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    elif name == "cifar100":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    return transforms.ToTensor()


def load_torchvision_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True,
) -> Dataset:
    """Download and return a torchvision dataset."""
    name = dataset_name.lower()
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from {list(_DATASET_REGISTRY.keys())}")
    cls, _, _ = _DATASET_REGISTRY[name]
    tf = _get_transforms(name)
    return cls(root=data_dir, train=train, download=True, transform=tf)


def get_input_channels(dataset_name: str) -> int:
    """Return number of input channels for a dataset."""
    return _DATASET_REGISTRY[dataset_name.lower()][1]


def get_num_classes(dataset_name: str) -> int:
    """Return number of classes for a dataset."""
    return _DATASET_REGISTRY[dataset_name.lower()][2]


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = SEED,
) -> Dict[int, List[int]]:
    """
    Split dataset indices IID uniformly across clients.

    Returns:
        Dict mapping client_id → list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {i: splits[i].tolist() for i in range(num_clients)}


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int = SEED,
) -> Dict[int, List[int]]:
    """
    Non-IID partition using Dirichlet distribution (α).

    Lower α → more heterogeneous. α=IID equivalent is a large number.

    Args:
        dataset: PyTorch dataset with a .targets attribute.
        num_clients: Number of FL clients.
        alpha: Dirichlet concentration parameter.
        seed: RNG seed.

    Returns:
        Dict mapping client_id → list of dataset indices.
    """
    rng = np.random.default_rng(seed)

    # Get labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


def get_partition(
    dataset: Dataset,
    num_clients: int,
    partition: str = "iid",
    alpha: float = 0.5,
    seed: int = SEED,
) -> Dict[int, List[int]]:
    """
    Partition a dataset for federated learning.

    Args:
        dataset: Full training dataset.
        num_clients: Number of FL clients.
        partition: 'iid' or 'dirichlet' (non-iid).
        alpha: Dirichlet alpha (used only when partition='dirichlet').
        seed: Random seed.

    Returns:
        Dict client_id → list of indices.
    """
    if partition.lower() == "iid":
        return iid_partition(dataset, num_clients, seed)
    elif partition.lower() in ("dirichlet", "non-iid", "noniid"):
        return dirichlet_partition(dataset, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partition '{partition}'. Use 'iid' or 'dirichlet'.")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_client_dataloaders(
    dataset: Dataset,
    client_indices: Dict[int, List[int]],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[int, DataLoader]:
    """
    Create a DataLoader per client from their assigned indices.

    Returns:
        Dict client_id → DataLoader.
    """
    loaders = {}
    for cid, idxs in client_indices.items():
        subset = Subset(dataset, idxs)
        loaders[cid] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders


def make_global_test_loader(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 256,
) -> DataLoader:
    """Return a DataLoader for the held-out global test set."""
    test_dataset = load_torchvision_dataset(dataset_name, data_dir, train=False)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

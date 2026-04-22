"""
Toy federated linear regression dataset.

Matches the paper's toy experiment (Appendix B.3):
  - 2 clients, each with 1024 samples and d=11 features
  - 9 numerical features ~ Uniform[0, 1)
  - 1 binary sensitive feature ~ Bernoulli(0.5), mapped to {-1, +1}
  - Labels: y = X @ theta* + eps, eps ~ N(0, 0.1)
  - theta* ~ N(0, 1)^d drawn fresh per client

The sensitive attribute is the last feature (index d-1 = 10).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

# Paper Appendix B.3 defaults
_DEFAULT_N_CLIENTS = 2
_DEFAULT_N_SAMPLES = 1024          # train samples per client
_DEFAULT_N_TEST = 128              # test samples per client
_DEFAULT_N_NUMERICAL = 9
_DEFAULT_N_BINARY = 1             # the sensitive attribute
_DEFAULT_NOISE_STD = 0.1
_DEFAULT_SEED = 42


class ToyLinearDataset(Dataset):
    """PyTorch Dataset wrapping toy linear-regression data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class FederatedToyDataset:
    """
    Synthetic federated dataset for toy linear regression experiments.

    Replicates Appendix B.3 of the paper:
      - n_clients clients, each owning n_train_samples samples with d features
      - d = n_numerical_features + n_binary_features  (default: 9 + 1 = 10,
        but a bias column is appended → effective d passed to LinearModel = 11)
      - n_numerical_features features ~ Uniform[0, 1)
      - n_binary_features binary feature(s) ~ {-1, +1}
      - The sensitive attribute is the *last* feature (index d-1)
      - Each client has its own randomly drawn optimal model theta*_c ~ N(0,1)^d
      - Labels: y = X @ theta*_c + eps,  eps ~ N(0, noise_std)

    Note on the bias column:
      The paper uses d=11 for a dataset with 10 real features.  This is because
      LinearLayer (from the original repo) / LinearModel (our models.py) adds a
      bias term, so the effective parameter vector has d+1 = 11 entries.
      We keep the same convention: n_features=10 raw columns, but the model's
      Linear layer has input_dimension=10 which internally learns an extra bias,
      giving 11 parameters total — matching the paper's "d=11".

    Args:
        n_clients:           Number of federated clients.
        n_train_samples:     Training samples per client.
        n_test_samples:      Test samples per client.
        n_numerical_features: Numerical features per sample.
        n_binary_features:   Binary features per sample (last one is sensitive).
        noise_std:           Standard deviation of label noise.
        seed:                Random seed for reproducibility.
    """

    # The sensitive attribute is always the last column.
    SENSITIVE_ATTR_ID: int = _DEFAULT_N_NUMERICAL  # index 9 (0-based) for default config

    def __init__(
        self,
        n_clients: int = _DEFAULT_N_CLIENTS,
        n_train_samples: int = _DEFAULT_N_SAMPLES,
        n_test_samples: int = _DEFAULT_N_TEST,
        n_numerical_features: int = _DEFAULT_N_NUMERICAL,
        n_binary_features: int = _DEFAULT_N_BINARY,
        noise_std: float = _DEFAULT_NOISE_STD,
        seed: int = _DEFAULT_SEED,
    ):
        self.n_clients = n_clients
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.n_numerical_features = n_numerical_features
        self.n_binary_features = n_binary_features
        self.n_features = n_numerical_features + n_binary_features
        self.noise_std = noise_std
        self.seed = seed

        # The sensitive attribute is the last feature column
        self.sensitive_attr_id = n_numerical_features + n_binary_features - 1

        # Model input dimension equals the number of raw features.
        # The Linear layer adds a bias internally, so parameter count = n_features + 1.
        self.input_dim = self.n_features

        self._rng = np.random.default_rng(seed)
        self._generate_all_clients()

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _generate_client_data(
        self, client_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate train/test data for a single client.

        Each client draws its own optimal model theta*_c ~ N(0,1)^n_features.
        Features:
          - Numerical: Uniform[0,1)
          - Binary: Bernoulli(0.5) mapped to {-1, +1}   (paper convention)
        Labels:  y = X @ theta*_c + N(0, noise_std)
        """
        n_total = self.n_train_samples + self.n_test_samples

        # Client-specific optimal model (used to generate labels)
        theta_c = self._rng.standard_normal(size=self.n_features).astype(np.float32)

        # Numerical features: Uniform[0, 1)
        numerical = self._rng.uniform(0.0, 1.0, size=(n_total, self.n_numerical_features)).astype(np.float32)

        # Binary features: {-1, +1}  (paper: "binary data sampled from Bernoulli, scaled to ±1")
        binary = self._rng.integers(0, 2, size=(n_total, self.n_binary_features)).astype(np.float32)
        binary = 2.0 * binary - 1.0  # map 0→-1, 1→+1

        features = np.concatenate([numerical, binary], axis=1)  # (n_total, n_features)

        # Labels: linear model + noise
        noise = self._rng.normal(0.0, self.noise_std, size=n_total).astype(np.float32)
        labels = features @ theta_c + noise  # (n_total,)

        train_features = features[: self.n_train_samples]
        train_labels = labels[: self.n_train_samples]
        test_features = features[self.n_train_samples :]
        test_labels = labels[self.n_train_samples :]

        return train_features, train_labels, test_features, test_labels

    def _generate_all_clients(self) -> None:
        """Generate and store data for every client."""
        self._train_datasets: List[ToyLinearDataset] = []
        self._test_datasets: List[ToyLinearDataset] = []

        for cid in range(self.n_clients):
            tr_X, tr_y, te_X, te_y = self._generate_client_data(cid)
            self._train_datasets.append(ToyLinearDataset(tr_X, tr_y))
            self._test_datasets.append(ToyLinearDataset(te_X, te_y))

        logger.info(
            f"FederatedToyDataset: {self.n_clients} clients, "
            f"{self.n_train_samples} train / {self.n_test_samples} test per client, "
            f"input_dim={self.input_dim}, sensitive_attr_id={self.sensitive_attr_id}"
        )

    # ------------------------------------------------------------------
    # Public interface (matches FederatedMedicalCostDataset / FederatedIncomeDataset API)
    # ------------------------------------------------------------------

    def num_clients(self) -> int:
        return self.n_clients

    def get_dataset(self, client_id: int, mode: str = "train") -> ToyLinearDataset:
        """Return the Dataset for a given client and mode ('train' or 'test')."""
        if mode == "train":
            return self._train_datasets[client_id]
        elif mode == "test":
            return self._test_datasets[client_id]
        else:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

    def get_dataloader(
        self,
        client_id: int,
        mode: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Return a DataLoader for a given client and mode."""
        dataset = self.get_dataset(client_id, mode)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

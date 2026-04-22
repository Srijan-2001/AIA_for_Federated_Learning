"""
Medical Cost dataset loader.

Dataset: Kaggle Medical Cost Personal Dataset
  - 1,339 records, 6 features: age, sex, bmi, children, smoker, region
  - Regression task: predict medical charges billed by health insurance
  - Sensitive attribute: smoker (binary: 0=no, 1=yes)
  - Split: 2 clients (i.i.d.)

Paper Section 5.1: "The dataset is split i.i.d. between 2 clients."
"""

import os
import io
import zipfile
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
TARGET_COLUMN = "charges"
SENSITIVE_ATTR = "smoker_yes"  # after one-hot encoding


class MedicalCostDataset(Dataset):
    """
    PyTorch Dataset for Medical Cost data.

    Args:
        features: numpy array of shape (n_samples, n_features).
        targets: numpy array of shape (n_samples,).
        column_names: list of feature column names.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, column_names: List[str]):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.column_names = column_names
        self.column_name_to_id = {name: i for i, name in enumerate(column_names)}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

    @property
    def sensitive_attr_id(self) -> int:
        """Index of the sensitive attribute (smoker_yes)."""
        return self.column_name_to_id[SENSITIVE_ATTR]


class FederatedMedicalCostDataset:
    """
    Prepares the Medical Cost dataset for federated learning.

    Handles:
      - Loading from ZIP file (uploaded dataset)
      - Preprocessing: one-hot encode categoricals, StandardScaler on numerics
      - Splitting into n_clients i.i.d. partitions (train/test)

    Args:
        data_path: Path to Medical_Cost.zip or medical_cost.csv.
        n_clients: Number of federated clients. Default 2 (as in paper).
        test_frac: Fraction of data held out for testing. Default 0.1 (paper uses 10%).
        seed: Random seed.
        scale_target: Whether to standardize the target (charges) column.
    """

    def __init__(
        self,
        data_path: str,
        n_clients: int = 2,
        test_frac: float = 0.1,
        seed: int = 42,
        scale_target: bool = True,
    ):
        self.data_path = data_path
        self.n_clients = n_clients
        self.test_frac = test_frac
        self.seed = seed
        self.scale_target = scale_target

        self.rng = np.random.default_rng(seed)
        self._scaler = StandardScaler()
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

        self._train_dfs: List[pd.DataFrame] = []
        self._test_dfs: List[pd.DataFrame] = []
        self.column_names: List[str] = []

        self._load_and_split()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        """Read the CSV from a zip file or directly."""
        if self.data_path.endswith(".zip"):
            with zipfile.ZipFile(self.data_path, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise FileNotFoundError(f"No CSV found inside {self.data_path}")
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(io.TextIOWrapper(f), sep=r"\s*,\s*", engine="python")
        else:
            df = pd.read_csv(self.data_path)
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        df = df.dropna().reset_index(drop=True)
        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=np.float64)
        return df

    def _scale(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit scaler on train numerics, transform both splits."""
        num_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
        self._scaler.fit(train_df[num_cols])
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[num_cols] = self._scaler.transform(train_df[num_cols])
        test_df[num_cols] = self._scaler.transform(test_df[num_cols])

        if self.scale_target:
            self._target_mean = train_df[TARGET_COLUMN].mean()
            self._target_std = train_df[TARGET_COLUMN].std()
            train_df[TARGET_COLUMN] = (train_df[TARGET_COLUMN] - self._target_mean) / (self._target_std + 1e-8)
            test_df[TARGET_COLUMN] = (test_df[TARGET_COLUMN] - self._target_mean) / (self._target_std + 1e-8)

        return train_df, test_df

    def _iid_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe i.i.d. across n_clients."""
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        splits = np.array_split(df, self.n_clients)
        return [s.reset_index(drop=True) for s in splits]

    def _load_and_split(self) -> None:
        logger.info(f"Loading Medical Cost dataset from {self.data_path}")
        raw = self._read_csv()
        processed = self._preprocess(raw)

        train_df, test_df = train_test_split(
            processed, test_size=self.test_frac, random_state=self.seed
        )
        train_df, test_df = self._scale(
            train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        )

        self._train_dfs = self._iid_split(train_df)
        self._test_dfs = self._iid_split(test_df)

        feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
        self.column_names = feature_cols
        self.input_dim = len(feature_cols)
        logger.info(
            f"Medical Cost: {self.n_clients} clients | "
            f"features={self.input_dim} | "
            f"sensitive='{SENSITIVE_ATTR}' (idx={self.column_names.index(SENSITIVE_ATTR)})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self, client_id: int, mode: str = "train") -> MedicalCostDataset:
        """
        Return PyTorch Dataset for client ``client_id``.

        Args:
            client_id: 0-based client index.
            mode: 'train' or 'test'.
        """
        assert mode in ("train", "test"), f"mode must be 'train' or 'test', got '{mode}'"
        df = self._train_dfs[client_id] if mode == "train" else self._test_dfs[client_id]
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        features = df[feature_cols].values
        targets = df[TARGET_COLUMN].values
        return MedicalCostDataset(features, targets, feature_cols)

    def get_dataloader(
        self,
        client_id: int,
        mode: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = self.get_dataset(client_id, mode)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and mode == "train"))

    def num_clients(self) -> int:
        return self.n_clients

    @property
    def sensitive_attr_name(self) -> str:
        return SENSITIVE_ATTR

    @property
    def sensitive_attr_id(self) -> int:
        return self.column_names.index(SENSITIVE_ATTR)

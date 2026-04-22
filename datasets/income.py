"""
ACS Income dataset loader (Income-L and Income-A scenarios from the paper).

Dataset: ACS Income (Ding et al. 2024) — folktables / ACSIncome_state_number.arff
  - Census data from 50 US states + Puerto Rico, 2014-2018
  - 15 features: age, occupation, education, etc.
  - Regression task: predict individual income
  - Sensitive attribute: SEX (binary: 1=male, 2=female → mapped to 0/1)

Paper Section 5.1:
  Income-L: 10 clients, Louisiana only, variable heterogeneity levels
  Income-A: 51 clients, one per census region, 20% random sample each
"""

import os
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Column names in the ARFF file (ACSIncome)
ARFF_FEATURE_COLS = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP",
    "SEX", "RAC1P", "PINCP", "ST",
]
# After loading: PINCP is the target, ST is the state code, SEX is sensitive
TARGET_COLUMN = "PINCP"
SENSITIVE_ATTR = "SEX"
STATE_COLUMN = "ST"

# Louisiana FIPS code in ACS data
LOUISIANA_CODE = 22  # ST=22

# Number of unique state codes (for Income-A, 51 partitions)
NUM_STATES = 51


class IncomeDataset(Dataset):
    """PyTorch Dataset for ACS Income data."""

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
        return self.column_name_to_id[SENSITIVE_ATTR]


class FederatedIncomeDataset:
    """
    Prepares the ACS Income dataset for federated learning.

    Supports two scenarios:
      - Income-L: 10 clients from a single state (Louisiana), with controllable
                  heterogeneity (Algorithm 5 in the paper).
      - Income-A: 51 clients, one per census region (state), 20% random sample.

    Args:
        data_path: Path to ACSIncome_state_number.arff.
        scenario: 'income_L' or 'income_A'.
        state_code: FIPS code for Income-L (22 = Louisiana). Ignored for Income-A.
        n_clients: Number of clients for Income-L. Default 10.
        heterogeneity: Heterogeneity level h ∈ [0, 0.5] for Income-L. Default 0.4.
        test_frac: Fraction of data for testing. Default 0.1.
        sample_frac: Fraction of state data each client uses for Income-A. Default 0.2.
        seed: Random seed.
        scale_target: Standardize income target.
    """

    def __init__(
        self,
        data_path: str,
        scenario: str = "income_L",
        state_code: int = LOUISIANA_CODE,
        n_clients: int = 10,
        heterogeneity: float = 0.4,
        test_frac: float = 0.1,
        sample_frac: float = 0.2,
        seed: int = 42,
        scale_target: bool = True,
    ):
        assert scenario in ("income_L", "income_A"), \
            f"scenario must be 'income_L' or 'income_A', got '{scenario}'"
        assert 0.0 <= heterogeneity <= 0.5, "heterogeneity must be in [0, 0.5]"

        self.data_path = data_path
        self.scenario = scenario
        self.state_code = state_code
        self.n_clients = n_clients
        self.heterogeneity = heterogeneity
        self.test_frac = test_frac
        self.sample_frac = sample_frac
        self.seed = seed
        self.scale_target = scale_target

        self.rng = np.random.default_rng(seed)
        self._scaler = StandardScaler()
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

        self._train_datasets: List[IncomeDataset] = []
        self._test_datasets: List[IncomeDataset] = []
        self.column_names: List[str] = []
        self.input_dim: int = 0
        self._actual_n_clients: int = 0

        self._load_and_split()

    # ------------------------------------------------------------------
    # ARFF loading
    # ------------------------------------------------------------------

    def _load_arff(self) -> pd.DataFrame:
        """Load the ARFF file into a pandas DataFrame."""
        logger.info(f"Loading ARFF file: {self.data_path}")
        data, meta = arff.loadarff(self.data_path)
        df = pd.DataFrame(data)

        # Decode bytes columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

        # Ensure numeric columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess ACS Income dataframe.
        - Binarize SEX: 1→1 (male), 2→0 (female) so sensitive attr is in {0,1}
        - Keep ST as int for splitting, then drop it from features
        """
        df = df.copy()
        # Map SEX: 1=male→1, 2=female→0
        df[SENSITIVE_ATTR] = (df[SENSITIVE_ATTR] == 1).astype(np.float32)
        # Clip negative income
        df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0)
        return df

    def _scale(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit scaler on training features, transform both splits."""
        train_df = train_df.copy()
        test_df = test_df.copy()

        self._scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = self._scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = self._scaler.transform(test_df[feature_cols])

        if self.scale_target:
            self._target_mean = train_df[TARGET_COLUMN].mean()
            self._target_std = train_df[TARGET_COLUMN].std()
            train_df[TARGET_COLUMN] = (
                (train_df[TARGET_COLUMN] - self._target_mean) / (self._target_std + 1e-8)
            )
            test_df[TARGET_COLUMN] = (
                (test_df[TARGET_COLUMN] - self._target_mean) / (self._target_std + 1e-8)
            )

        return train_df, test_df

    # ------------------------------------------------------------------
    # Income-L heterogeneity split (Algorithm 5 from the paper)
    # ------------------------------------------------------------------

    def _heterogeneous_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split a state's dataframe into n_clients with heterogeneity level h.

        IMPORTANT: This must be called on UNSCALED data so that SEX is still
        binary {0, 1}. StandardScaler transforms SEX to ~{-1.0, 0.98} which
        breaks the (SEX == 1) and (SEX == 0) conditions below.

        Algorithm 5 (paper):
          1. Partition into D_h (rich men + poor women) and D_l (poor men + rich women)
          2. Balance sizes, then randomly swap (0.5 - h) * k samples between clusters
          3. Assign D_h' to clients 5-9, D_l' to clients 0-4
        """
        h = self.heterogeneity
        med = df[TARGET_COLUMN].median()

        # D_h: rich men OR poor women
        D_h = df[
            ((df[SENSITIVE_ATTR] == 1) & (df[TARGET_COLUMN] > med)) |
            ((df[SENSITIVE_ATTR] == 0) & (df[TARGET_COLUMN] <= med))
        ].copy()

        # D_l: poor men OR rich women (complement)
        D_l = df[~df.index.isin(D_h.index)].copy()

        k = min(len(D_h), len(D_l))
        D_h = D_h.sample(k, random_state=self.seed).reset_index(drop=True)
        D_l = D_l.sample(k, random_state=self.seed).reset_index(drop=True)

        # Swap (0.5 - h) * k samples between the two clusters
        swap_n = int((0.5 - h) * k)
        if swap_n > 0:
            swap_h = D_h.sample(swap_n, random_state=self.seed).index
            swap_l = D_l.sample(swap_n, random_state=self.seed).index

            tmp_h = D_h.loc[swap_h].copy()
            tmp_l = D_l.loc[swap_l].copy()

            D_h_new = pd.concat(
                [D_h.drop(swap_h), tmp_l], ignore_index=True
            )
            D_l_new = pd.concat(
                [D_l.drop(swap_l), tmp_h], ignore_index=True
            )
            D_h = D_h_new
            D_l = D_l_new

        half = self.n_clients // 2
        splits_l = np.array_split(D_l.sample(frac=1, random_state=self.seed), half)
        splits_h = np.array_split(D_h.sample(frac=1, random_state=self.seed), self.n_clients - half)

        all_splits = [pd.DataFrame(s).reset_index(drop=True) for s in splits_l + splits_h]

        # Filter out empty shards that occur when the dataset is small relative to n_clients
        non_empty = [s for s in all_splits if len(s) > 0]
        if len(non_empty) < len(all_splits):
            logger.warning(
                f"_heterogeneous_split: {len(all_splits) - len(non_empty)} empty shard(s) "
                f"dropped (dataset too small for {self.n_clients} clients). "
                f"Effective client count reduced to {len(non_empty)}."
            )
        if len(non_empty) == 0:
            raise ValueError(
                "All client shards are empty after heterogeneous split. "
                "The dataset is too small for the requested number of clients."
            )
        return non_empty

    # ------------------------------------------------------------------
    # Income-A: one client per state, 20% random sample
    # ------------------------------------------------------------------

    def _state_split(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        For Income-A: split dataframe by state, sample 20% per state.

        Returns:
            dict mapping state_code → DataFrame
        """
        state_dfs = {}
        for code, group in df.groupby(STATE_COLUMN):
            sampled = group.sample(
                frac=self.sample_frac, random_state=self.seed
            ).reset_index(drop=True)
            state_dfs[int(code)] = sampled
        return state_dfs

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _df_to_dataset(self, df: pd.DataFrame, feature_cols: List[str]) -> IncomeDataset:
        features = df[feature_cols].values
        targets = df[TARGET_COLUMN].values
        return IncomeDataset(features, targets, feature_cols)

    def _load_and_split(self) -> None:
        raw = self._load_arff()
        df = self._preprocess(raw)

        if self.scenario == "income_L":
            df_state = df[df[STATE_COLUMN] == self.state_code].reset_index(drop=True)
            if len(df_state) == 0:
                raise ValueError(
                    f"No records for state_code={self.state_code}. "
                    f"Available: {sorted(df[STATE_COLUMN].unique())}"
                )
            feature_cols = [c for c in df_state.columns if c not in (TARGET_COLUMN, STATE_COLUMN)]

            train_df, test_df = train_test_split(
                df_state, test_size=self.test_frac, random_state=self.seed
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            # IMPORTANT: split BEFORE scaling so SEX is still binary {0,1}
            # when _heterogeneous_split checks (SEX == 1) and (SEX == 0).
            # Scaling transforms SEX to ~{-1.0, 0.98} which breaks the conditions.
            client_train_dfs = self._heterogeneous_split(train_df)
            actual_n = len(client_train_dfs)

            client_test_dfs = np.array_split(
                test_df.sample(frac=1, random_state=self.seed),
                actual_n,
            )

            # Fit scaler on full unscaled training data, then apply shard by shard
            self._scaler.fit(train_df[feature_cols])
            if self.scale_target:
                self._target_mean = train_df[TARGET_COLUMN].mean()
                self._target_std = train_df[TARGET_COLUMN].std()

            def _apply_scale(shard: pd.DataFrame) -> pd.DataFrame:
                shard = shard.copy()
                shard[feature_cols] = self._scaler.transform(shard[feature_cols])
                if self.scale_target:
                    shard[TARGET_COLUMN] = (
                        (shard[TARGET_COLUMN] - self._target_mean)
                        / (self._target_std + 1e-8)
                    )
                return shard

            self._train_datasets = [
                self._df_to_dataset(_apply_scale(d), feature_cols)
                for d in client_train_dfs
            ]
            self._test_datasets = [
                self._df_to_dataset(_apply_scale(pd.DataFrame(d).reset_index(drop=True)), feature_cols)
                for d in client_test_dfs
            ]
            self._actual_n_clients = actual_n

        else:  # income_A
            feature_cols = [c for c in df.columns if c not in (TARGET_COLUMN, STATE_COLUMN)]

            train_df, test_df = train_test_split(
                df, test_size=self.test_frac, random_state=self.seed
            )
            train_df, test_df = self._scale(
                train_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                feature_cols,
            )

            train_state_dfs = self._state_split(train_df)
            test_state_dfs = self._state_split(test_df)

            state_codes = sorted(train_state_dfs.keys())
            self._state_codes = state_codes
            self._actual_n_clients = len(state_codes)

            self._train_datasets = [
                self._df_to_dataset(train_state_dfs[code], feature_cols)
                for code in state_codes
            ]
            self._test_datasets = [
                self._df_to_dataset(test_state_dfs.get(code, train_state_dfs[code]), feature_cols)
                for code in state_codes
            ]

        self.column_names = feature_cols
        self.input_dim = len(feature_cols)

        logger.info(
            f"Income ({self.scenario}): {self._actual_n_clients} clients | "
            f"features={self.input_dim} | sensitive='{SENSITIVE_ATTR}' "
            f"(idx={feature_cols.index(SENSITIVE_ATTR)})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self, client_id: int, mode: str = "train") -> IncomeDataset:
        assert mode in ("train", "test")
        if mode == "train":
            return self._train_datasets[client_id]
        return self._test_datasets[client_id]

    def get_dataloader(
        self,
        client_id: int,
        mode: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        ds = self.get_dataset(client_id, mode)
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and mode == "train"))

    def num_clients(self) -> int:
        return self._actual_n_clients

    @property
    def sensitive_attr_name(self) -> str:
        return SENSITIVE_ATTR

    @property
    def sensitive_attr_id(self) -> int:
        return self.column_names.index(SENSITIVE_ATTR)
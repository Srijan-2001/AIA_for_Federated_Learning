"""
main.py — End-to-end FL + AIA pipeline using Flower (flwr).

This script:
  1. Loads a real-world dataset (Medical Cost or ACS Income).
  2. Initialises Flower clients (one per federated party).
  3. Runs FedAvg training via flwr.simulation (in-process simulation).
  4. Runs Attribute Inference Attacks at the end:
       - Our passive model-based AIA  (Algorithm 2 + Eq. 3)
       - Our active model-based AIA   (Algorithm 3 + Eq. 3)
       - Gradient-based AIA baseline  (Lyu & Chen 2021)
       - Global-model AIA baseline
  5. Saves results to JSON.

Usage examples
--------------
# Medical Cost (2 clients, passive attack, infer smoking status)
python main.py --dataset medical_cost --data_path data/Medical_Cost.zip \
    --num_rounds 100 --attack passive

# ACS Income-L (10 clients, Louisiana, active attack, infer gender)
python main.py --dataset income_L --data_path data/ACSIncome_state_number.arff \
    --num_rounds 100 --num_clients 10 --heterogeneity 0.4 \
    --attack active --active_rounds 50

# ACS Income-A (51 clients, all states, active attack)
python main.py --dataset income_A --data_path data/ACSIncome_state_number.arff \
    --num_rounds 100 --attack active --active_rounds 50

# Reproduce Table 1 (neural network, all 3 datasets)
python main.py --reproduce_table1 --data_dir data/
"""

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.simulation import start_simulation

from models import get_model, get_flat_params, set_flat_params
from fl_client import FedAvgClient
from fl_server import FedAvgWithAIA
from utils import (
    configure_logging, set_seed, save_results,
    save_results_csv, save_multi_seed_csv,
    plot_accuracy_loss_curves, plot_aia_comparison,
)
from datasets.medical_cost import FederatedMedicalCostDataset
from datasets.income import FederatedIncomeDataset

logger = logging.getLogger(__name__)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attribute Inference Attacks for Federated Regression Tasks — Flower",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument(
        "--dataset",
        type=str,
        default="medical_cost",
        choices=["medical_cost", "income_L", "income_A"],
        help="Dataset / FL scenario to use.",
    )
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to the dataset file (zip or arff).")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Directory containing dataset files (used with --reproduce_table1).")

    # Income-L specific
    p.add_argument("--state_code", type=int, default=22,
                   help="FIPS state code for Income-L scenario (22=Louisiana).")
    p.add_argument("--heterogeneity", type=float, default=0.4,
                   help="Heterogeneity level h ∈ [0,0.5] for Income-L split.")
    p.add_argument("--sample_frac", type=float, default=0.2,
                   help="Fraction of state data sampled per client (Income-A).")

    # FL training
    p.add_argument("--num_rounds", type=int, default=100,
                   help="Number of FL communication rounds.")
    p.add_argument("--num_clients", type=int, default=None,
                   help="Number of clients (overrides dataset default).")
    p.add_argument("--local_epochs", type=int, default=1,
                   help="Number of local training epochs per round (E in paper).")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Local training batch size.")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Client SGD learning rate. Auto-set per dataset if None.")
    p.add_argument("--test_frac", type=float, default=0.1,
                   help="Fraction of data used for testing.")
    p.add_argument("--scale_target", action="store_true", default=True,
                   help="Standardise target variable.")

    # Model
    p.add_argument("--model_type", type=str, default="neural_network",
                   choices=["neural_network", "linear"],
                   help="Model architecture.")
    p.add_argument("--hidden_size", type=int, default=128,
                   help="Hidden layer size for neural network.")

    # Attack
    p.add_argument("--attack", type=str, default="passive",
                   choices=["passive", "active", "gradient", "none"],
                   help="Attack mode.")
    p.add_argument("--targeted_client", type=int, default=0,
                   help="ID (0-based index) of the targeted client.")
    p.add_argument("--active_start_round", type=int, default=None,
                   help="Round at which active attack begins. Default: num_rounds - active_rounds.")
    p.add_argument("--active_rounds", type=int, default=50,
                   help="Number of active attack rounds.")
    p.add_argument("--adam_lr", type=float, default=1.0,
                   help="Adam learning rate for active attack.")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--grad_aia_iters", type=int, default=5000,
                   help="Gradient-based AIA optimisation iterations.")
    p.add_argument("--max_grad_rounds", type=int, default=10,
                   help="Max eavesdropped rounds used per gradient-AIA iteration "
                        "(sub-sampled evenly from all rounds to keep cost bounded).")

    # Output
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--seed", type=int, default=42)
    # Item 12: multi-seed support — runs the experiment for each seed and reports mean ± std
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run experiment for each seed and report mean ± std. "
             "Overrides --seed when set. Example: --seeds 42 43 44",
    )
    p.add_argument("--device", type=str, default="cpu")

    # Reproduction
    p.add_argument("--reproduce_table1", action="store_true",
                   help="Reproduce Table 1 from the paper (runs all 3 datasets).")

    return p.parse_args(argv)


# ===========================================================================
# Dataset loading
# ===========================================================================

def load_dataset(args: argparse.Namespace):
    """Load and return the federated dataset object."""
    if args.dataset == "medical_cost":
        data_path = args.data_path or os.path.join(args.data_dir, "Medical_Cost.zip")
        n_clients = args.num_clients or 2
        fed_dataset = FederatedMedicalCostDataset(
            data_path=data_path,
            n_clients=n_clients,
            test_frac=args.test_frac,
            seed=args.seed,
            scale_target=args.scale_target,
        )
        default_lr = 2e-6

    elif args.dataset == "income_L":
        data_path = args.data_path or os.path.join(args.data_dir, "ACSIncome_state_number.arff")
        n_clients = args.num_clients or 10
        fed_dataset = FederatedIncomeDataset(
            data_path=data_path,
            scenario="income_L",
            state_code=args.state_code,
            n_clients=n_clients,
            heterogeneity=args.heterogeneity,
            test_frac=args.test_frac,
            seed=args.seed,
            scale_target=args.scale_target,
        )
        default_lr = 5e-7

    elif args.dataset == "income_A":
        data_path = args.data_path or os.path.join(args.data_dir, "ACSIncome_state_number.arff")
        fed_dataset = FederatedIncomeDataset(
            data_path=data_path,
            scenario="income_A",
            sample_frac=args.sample_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            scale_target=args.scale_target,
        )
        default_lr = 1e-6

    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    lr = args.learning_rate if args.learning_rate is not None else default_lr
    return fed_dataset, lr


# ===========================================================================
# Client factory for flwr.simulation
# ===========================================================================

def make_client_fn(
    fed_dataset,
    model_init_fn,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: str,
    device: str,
):
    """Return a client_fn suitable for flwr.simulation.start_simulation."""

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        train_loader = fed_dataset.get_dataloader(client_id, mode="train",
                                                   batch_size=batch_size, shuffle=True)
        test_loader = fed_dataset.get_dataloader(client_id, mode="test",
                                                  batch_size=batch_size, shuffle=False)
        model = model_init_fn()

        client = FedAvgClient(
            client_id=cid,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=True,
        )
        return client

    return client_fn


# ===========================================================================
# Run a single FL + AIA experiment
# ===========================================================================

def run_experiment(args: argparse.Namespace) -> Dict:
    """
    Run one complete FL training + AIA experiment.

    Returns:
        Dictionary of AIA accuracy results.
    """
    set_seed(args.seed)
    configure_logging(args.log_level)

    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset} | Attack: {args.attack} | "
                f"Rounds: {args.num_rounds} | Local epochs: {args.local_epochs}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    fed_dataset, lr = load_dataset(args)
    n_clients = fed_dataset.num_clients()
    input_dim = fed_dataset.input_dim
    sensitive_attr_id = fed_dataset.sensitive_attr_id
    targeted_cid = str(args.targeted_client)

    logger.info(f"Clients: {n_clients} | Input dim: {input_dim} | "
                f"Sensitive attr idx: {sensitive_attr_id} | LR: {lr}")

    # ------------------------------------------------------------------
    # 2. Model factory
    # ------------------------------------------------------------------
    def model_init_fn():
        return get_model(
            model_type=args.model_type,
            input_dimension=input_dim,
            output_dimension=1,
            hidden_layers=[args.hidden_size] if args.model_type == "neural_network" else None,
        )

    # ------------------------------------------------------------------
    # 3. Initial parameters
    # ------------------------------------------------------------------
    init_model = model_init_fn()
    init_params = ndarrays_to_parameters(
        [p.data.cpu().numpy() for p in init_model.parameters()]
    )

    # ------------------------------------------------------------------
    # 4. Build strategy with AIA hooks
    # ------------------------------------------------------------------
    active_start = (
        args.active_start_round
        if args.active_start_round is not None
        else max(1, args.num_rounds - args.active_rounds)
    )

    strategy = FedAvgWithAIA(
        model_init_fn=model_init_fn,
        initial_parameters=init_params,
        targeted_client_id=targeted_cid,
        attack_mode=args.attack if args.attack != "none" else "passive",
        active_start_round=active_start,
        active_rounds=args.active_rounds,
        adam_lr=args.adam_lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.dataset),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 5. Build client_fn
    # ------------------------------------------------------------------
    client_fn = make_client_fn(
        fed_dataset=fed_dataset,
        model_init_fn=model_init_fn,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=lr,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.dataset),
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 6. Run FL simulation via Flower
    # ------------------------------------------------------------------
    logger.info("Starting Flower FL simulation...")
    history = start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    logger.info("FL simulation complete.")
    logger.info(f"Final global model loss (distributed eval): "
                f"{history.losses_distributed[-1][1]:.4f}" if history.losses_distributed else "N/A")

    # ------------------------------------------------------------------
    # 7. Run AIA
    # ------------------------------------------------------------------
    if args.attack == "none":
        return {"fl_history": str(history)}

    targeted_train_dataset = fed_dataset.get_dataset(args.targeted_client, mode="train")

    # FIX: pass nn_model_type so run_aia() selects the correct passive reconstruction
    # strategy — LMRA for linear models, last returned local model for neural networks.
    aia_results = strategy.run_aia(
        targeted_dataset=targeted_train_dataset,
        sensitive_attr_id=sensitive_attr_id,
        nn_model_type=args.model_type,
        num_grad_aia_iterations=args.grad_aia_iters,
        max_grad_rounds=args.max_grad_rounds,
    )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    os.makedirs(args.results_dir, exist_ok=True)
    result_path = os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.attack}_r{args.num_rounds}_seed{args.seed}.json",
    )
    # Keys that are accuracy fractions in [0, 1] and should be scaled to percent.
    # MSE values, loss, communication cost etc. must NOT be multiplied by 100.
    _ACCURACY_KEYS = {"ours", "global_model", "grad_passive", "final_accuracy"}
    full_results = {
        "config": vars(args),
        "aia_accuracy_percent": {
            k: round(v * 100, 2) if k in _ACCURACY_KEYS else round(v, 6)
            for k, v in aia_results.items()
            if isinstance(v, float)
        },
        "aia_accuracy_raw": {k: v for k, v in aia_results.items() if isinstance(v, float)},
        "n_clients": n_clients,
        "input_dim": input_dim,
        "sensitive_attr_id": sensitive_attr_id,
        "targeted_client": args.targeted_client,
        "learning_rate": lr,
        # Universal metrics (Items 1-4)
        "accuracy_history": aia_results.get("accuracy_history", []),
        "loss_history": aia_results.get("loss_history", []),
        "final_accuracy": aia_results.get("final_accuracy"),
        "final_loss": aia_results.get("final_loss"),
        "convergence_round": aia_results.get("convergence_round"),
        "communication_cost_mb": aia_results.get("communication_cost_mb"),
    }
    save_results(full_results, result_path)

    # ── Item 8: also save a flat CSV row ──────────────────────────────────
    csv_row = {
        "dataset": args.dataset,
        "attack": args.attack,
        "seed": args.seed,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "ours_pct": round(aia_results.get("ours", float("nan")) * 100, 2),
        "global_model_pct": round(aia_results.get("global_model", float("nan")) * 100, 2),
        "grad_passive_pct": round(aia_results.get("grad_passive", float("nan")) * 100, 2),
        "ours_mse": round(aia_results.get("ours_mse", float("nan")), 6),
        "global_model_mse": round(aia_results.get("global_model_mse", float("nan")), 6),
        "final_mse": round(aia_results.get("final_accuracy") or float("nan"), 6),
        "final_loss": aia_results.get("final_loss"),
        "convergence_round": aia_results.get("convergence_round"),
        "communication_cost_mb": aia_results.get("communication_cost_mb"),
    }
    csv_path = os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.attack}_r{args.num_rounds}_seed{args.seed}.csv",
    )
    save_results_csv([csv_row], csv_path)

    # ── Items 9-10: generate PNG figures ──────────────────────────────────
    figures_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Accuracy/loss curves (Item 9)
    acc_hist = aia_results.get("accuracy_history", [])
    loss_hist = aia_results.get("loss_history", [])
    if acc_hist or loss_hist:
        plot_accuracy_loss_curves(
            accuracy_history=acc_hist,
            loss_history=loss_hist,
            title=f"{args.dataset} | {args.attack} | seed {args.seed}",
            save_path=os.path.join(
                figures_dir,
                f"{args.dataset}_{args.attack}_r{args.num_rounds}_seed{args.seed}_curves.png",
            ),
        )

    # AIA comparison bar chart (Item 10)
    comparison_dict = {
        k: v for k, v in aia_results.items()
        if k in ("grad_passive", "ours", "global_model") and isinstance(v, float)
    }
    if comparison_dict:
        plot_aia_comparison(
            results_by_method=comparison_dict,
            dataset_name=args.dataset,
            attack_mode=args.attack,
            save_path=os.path.join(
                figures_dir,
                f"{args.dataset}_{args.attack}_r{args.num_rounds}_seed{args.seed}_aia_comparison.png",
            ),
        )

    return full_results


# ===========================================================================
# Table 1 reproduction
# ===========================================================================

def reproduce_table1(args: argparse.Namespace) -> None:
    """
    Reproduce Table 1 from the paper: AIA accuracy for neural networks
    on all three datasets (Income-L, Income-A, Medical) under passive
    and active attacks, matching the paper's table format exactly.

    Table 1 (paper results for reference):
      AIA (%)              | Income-L | Income-A | Medical
      ---------------------|----------|----------|--------
      Passive   Grad       |  60.36   |  54.98   |  87.26
                Ours       |  75.27   |  55.75   |  95.90
      Active    Grad       |  60.24   |  54.98   |  87.26
      (10 Rnds) Ours       |  82.02   |  63.53   |  95.93
      Active    Grad       |  60.24   |  53.36   |  87.26
      (50 Rnds) Ours       |  94.31   |  78.09   |  96.79
      Global Model         |  94.31   |  78.31   |  96.79
    """

    configs = [
        # (dataset, data_file, n_clients, lr, heterogeneity, attack, active_rounds, run_key)
        ("medical_cost", "Medical_Cost.zip",            2,    2e-6, 0.0, "passive", 0,  "medical_cost_passive"),
        ("medical_cost", "Medical_Cost.zip",            2,    2e-6, 0.0, "active",  10, "medical_cost_active10"),
        ("medical_cost", "Medical_Cost.zip",            2,    2e-6, 0.0, "active",  50, "medical_cost_active50"),
        ("income_L",     "ACSIncome_state_number.arff", 10,   5e-7, 0.4, "passive", 0,  "income_L_passive"),
        ("income_L",     "ACSIncome_state_number.arff", 10,   5e-7, 0.4, "active",  10, "income_L_active10"),
        ("income_L",     "ACSIncome_state_number.arff", 10,   5e-7, 0.4, "active",  50, "income_L_active50"),
        ("income_A",     "ACSIncome_state_number.arff", None, 1e-6, 0.0, "passive", 0,  "income_A_passive"),
        ("income_A",     "ACSIncome_state_number.arff", None, 1e-6, 0.0, "active",  10, "income_A_active10"),
        ("income_A",     "ACSIncome_state_number.arff", None, 1e-6, 0.0, "active",  50, "income_A_active50"),
    ]

    # Determine seeds to run over. If --seeds supplied, average results across
    # all provided seeds; otherwise fall back to the single --seed value.
    seeds_to_run = args.seeds if (args.seeds is not None and len(args.seeds) > 0) else [args.seed]

    all_results: Dict[str, Dict] = {}          # run_key → averaged aia_accuracy_percent
    all_results_per_seed: Dict[str, List] = {} # run_key → list of aia_accuracy_percent dicts

    for dataset, data_file, n_clients, lr, het, attack, active_rnds, run_key in configs:
        logger.info(f"\n{'='*70}\nRunning: {run_key}\n{'='*70}")

        per_seed_acc: List[Dict] = []
        for seed in seeds_to_run:
            logger.info(f"  Seed {seed} / {run_key}")
            run_args = deepcopy(args)
            run_args.dataset = dataset
            run_args.data_path = os.path.join(args.data_dir, data_file)
            run_args.num_clients = n_clients
            run_args.learning_rate = lr
            run_args.heterogeneity = het
            run_args.attack = attack
            run_args.active_rounds = active_rnds
            run_args.active_start_round = None  # let run_experiment compute it correctly
            run_args.num_rounds = 100
            run_args.local_epochs = 1
            run_args.batch_size = 32
            run_args.model_type = "neural_network"
            run_args.hidden_size = 128
            run_args.seed = seed
            run_args.seeds = None   # prevent run_experiment entering multi-seed branch

            results = run_experiment(run_args)
            per_seed_acc.append(results.get("aia_accuracy_percent", {}))

        # Average across seeds
        all_results_per_seed[run_key] = per_seed_acc
        if len(per_seed_acc) == 1:
            all_results[run_key] = per_seed_acc[0]
        else:
            # Compute mean over seeds for each metric key
            all_keys = set().union(*[d.keys() for d in per_seed_acc])
            avg: Dict = {}
            for k in all_keys:
                vals = [d[k] for d in per_seed_acc if isinstance(d.get(k), (int, float))]
                avg[k] = round(float(np.mean(vals)), 2) if vals else float("nan")
            all_results[run_key] = avg

    # -----------------------------------------------------------------------
    # Print table matching paper Table 1 format
    # -----------------------------------------------------------------------
    DATASETS  = ["income_L", "income_A", "medical_cost"]
    DS_LABELS = ["Income-L", "Income-A", "Medical"]
    COL_W = 14  # width per dataset column

    sep  = "+" + "-" * 22 + "+" + (("-" * COL_W + "+") * len(DATASETS))
    head = f"| {'AIA (%)':<20} |" + "".join(f" {lb:^{COL_W-2}} |" for lb in DS_LABELS)

    def fmt(val):
        return f"{val:.2f}" if not (isinstance(val, float) and val != val) else "N/A"

    def row(label1, label2, keys_by_ds, metric):
        vals = []
        for ds in DATASETS:
            key = keys_by_ds[ds]
            v = all_results.get(key, {}).get(metric, float("nan"))
            vals.append(fmt(v))
        col1 = f"{label1:<10}{label2:<12}"
        return "| " + col1 + "|" + "".join(f" {v:^{COL_W-2}} |" for v in vals)

    print("\n" + sep)
    print(head)
    print(sep)

    # Passive
    passive_keys = {ds: f"{ds}_passive" for ds in DATASETS}
    print(row("Passive",   "Grad", passive_keys, "grad_passive"))
    print(row("",          "Ours", passive_keys, "ours"))
    print(sep)

    # Active 10 rounds
    active10_keys = {ds: f"{ds}_active10" for ds in DATASETS}
    print(row("Active",    "Grad", active10_keys, "grad_passive"))
    print(row("(10 Rnds)", "Ours", active10_keys, "ours"))
    print(sep)

    # Active 50 rounds
    active50_keys = {ds: f"{ds}_active50" for ds in DATASETS}
    print(row("Active",    "Grad", active50_keys, "grad_passive"))
    print(row("(50 Rnds)", "Ours", active50_keys, "ours"))
    print(sep)

    # Global model — bottom reference row
    print(row("Global Model", "", passive_keys, "global_model"))
    print(sep)

    json_path = os.path.join(args.results_dir, "table1_reproduction.json")
    save_results(
        {"averaged": all_results, "seeds": seeds_to_run, "per_seed": all_results_per_seed},
        json_path,
    )

    # ── Save table1_reproduction as CSV too ──────────────────────────────
    # Build one flat CSV row per (dataset × attack_variant) combination so the
    # table can be loaded directly into pandas / spreadsheet tools.
    _TABLE1_ATTACK_VARIANTS = [
        ("passive",   0,  "passive"),
        ("active",    10, "active10"),
        ("active",    50, "active50"),
    ]
    csv_rows = []
    for dataset, ds_label in zip(DATASETS, DS_LABELS):
        # global_model baseline — take from the passive key (same model, run once)
        global_val = all_results.get(f"{dataset}_passive", {}).get("global_model", float("nan"))
        for attack, active_rnds, variant_suffix in _TABLE1_ATTACK_VARIANTS:
            run_key = f"{dataset}_{variant_suffix}"
            metrics = all_results.get(run_key, {})
            csv_rows.append({
                "dataset":           ds_label,
                "attack":            attack,
                "active_rounds":     active_rnds,
                "run_key":           run_key,
                "ours_pct":          metrics.get("ours",         float("nan")),
                "grad_passive_pct":  metrics.get("grad_passive", float("nan")),
                "global_model_pct":  global_val,
                "ours_mse":          metrics.get("ours_mse",          float("nan")),
                "global_model_mse":  metrics.get("global_model_mse",  float("nan")),
                "seeds":             str(seeds_to_run),
            })

    csv_path = os.path.join(args.results_dir, "table1_reproduction.csv")
    save_results_csv(csv_rows, csv_path)
    logger.info(f"Table 1 CSV saved to {csv_path}")


# ===========================================================================
# Multi-seed runner — Item 12: mean ± std over 3 seeds
# ===========================================================================

def run_multi_seed(args: argparse.Namespace) -> Dict:
    """
    Run run_experiment for each seed in args.seeds and aggregate mean ± std.

    Saves:
      - Per-seed JSON and CSV (via run_experiment)
      - A combined multi-seed CSV with mean ± std
      - A combined AIA comparison bar chart (mean values)

    Returns:
        Dict with per-seed results and aggregated stats.
    """
    seeds = args.seeds
    logger.info(f"Multi-seed run: seeds={seeds}")

    per_seed_aia: List[Dict] = []
    for seed in seeds:
        run_args = deepcopy(args)
        run_args.seed = seed
        run_args.seeds = None  # prevent infinite recursion
        logger.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        result = run_experiment(run_args)
        # Collect flat AIA metrics
        flat = {}
        for k, v in result.get("aia_accuracy_raw", {}).items():
            if isinstance(v, float):
                flat[k] = v
        flat["convergence_round"] = result.get("convergence_round")
        flat["communication_cost_mb"] = result.get("communication_cost_mb")
        flat["final_accuracy"] = result.get("final_accuracy")
        flat["final_loss"] = result.get("final_loss")
        per_seed_aia.append(flat)

    # Aggregate stats
    os.makedirs(args.results_dir, exist_ok=True)
    agg_csv_path = os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.attack}_r{args.num_rounds}_multiseed_summary.csv",
    )
    save_multi_seed_csv(per_seed_aia, agg_csv_path)

    # Compute means for the comparison plot
    all_keys = set().union(*[d.keys() for d in per_seed_aia])
    mean_results = {}
    std_results = {}
    for k in all_keys:
        vals = [d[k] for d in per_seed_aia if isinstance(d.get(k), (int, float)) and d.get(k) is not None]
        if vals:
            mean_results[k] = float(np.mean(vals))
            std_results[k] = float(np.std(vals))

    # Combined AIA comparison plot (Item 10) with mean values
    figures_dir = os.path.join(args.results_dir, "figures")
    comparison_dict = {
        k: v for k, v in mean_results.items()
        if k in ("grad_passive", "ours", "global_model")
    }
    if comparison_dict:
        plot_aia_comparison(
            results_by_method=comparison_dict,
            dataset_name=args.dataset,
            attack_mode=f"{args.attack} (mean over {len(seeds)} seeds)",
            save_path=os.path.join(
                figures_dir,
                f"{args.dataset}_{args.attack}_r{args.num_rounds}_multiseed_aia_comparison.png",
            ),
        )

    summary = {
        "seeds": seeds,
        "dataset": args.dataset,
        "attack": args.attack,
        "mean": mean_results,
        "std": std_results,
        "per_seed": per_seed_aia,
    }
    save_results(summary, os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.attack}_r{args.num_rounds}_multiseed_summary.json",
    ))

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Multi-seed summary ({args.dataset} | {args.attack})")
    print(f"{'='*60}")
    for k in ("grad_passive", "ours", "global_model"):
        if k in mean_results:
            print(f"  {k:<25s}: {mean_results[k]*100:.2f}% ± {std_results.get(k,0)*100:.2f}%")
    print(f"{'='*60}")

    return summary


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    args = parse_args()
    configure_logging(args.log_level)

    if args.reproduce_table1:
        reproduce_table1(args)
    elif args.seeds is not None:
        # Item 12: multi-seed run
        run_multi_seed(args)
    else:
        if args.data_path is None:
            # Auto-detect from data_dir
            default_files = {
                "medical_cost": "Medical_Cost.zip",
                "income_L": "ACSIncome_state_number.arff",
                "income_A": "ACSIncome_state_number.arff",
            }
            fname = default_files.get(args.dataset, "")
            candidate = os.path.join(args.data_dir, fname)
            if os.path.exists(candidate):
                args.data_path = candidate
            else:
                print(
                    f"\nERROR: data_path not specified and '{candidate}' not found.\n"
                    f"Please provide --data_path or place the dataset in '{args.data_dir}/'\n"
                    f"Expected files:\n"
                    f"  Medical Cost : {args.data_dir}/Medical_Cost.zip\n"
                    f"  ACS Income   : {args.data_dir}/ACSIncome_state_number.arff\n"
                )
                sys.exit(1)

        results = run_experiment(args)
        print("\nAIA Results:")
        for k, v in results.get("aia_accuracy_percent", {}).items():
            print(f"  {k:<25s}: {v:.2f}%")


if __name__ == "__main__":
    main()
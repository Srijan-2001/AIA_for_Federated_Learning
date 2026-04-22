# Federated Learning — Attribute Inference Attacks (Category 1: Privacy & Inference)

**Paper**: "Attribute Inference Attacks for Federated Regression Tasks"  
**Authors**: "Raju Debnath, Srijan Ghosh, Gunja Kumari, Ashwani, Ranjesh Kumar Roy, Shubham Anand"
**Framework**: [Flower (flwr)](https://flower.ai/)

---

## Group Info

| Field | Details |
|-------|---------|
| Category | Cat. 1 — Privacy & Inference |
| Paper | Diana et al. (AAAI 2025) |
| Framework | Flower (flwr) |

---

## Repository Structure

```
flower_aia_fl/
├── configs/
│   ├── baseline_fedavg.yaml          # FedAvg baseline config
│   └── aia_experiment.yaml           # AIA experiment config
├── attacks/
│   ├── __init__.py
│   ├── model_based_aia.py            # Model-based AIA (Eq. 3 + Alg. 2 + Alg. 3)
│   └── gradient_based_aia.py         # Gradient-based AIA baseline (Lyu & Chen 2021)
├── datasets/
│   ├── __init__.py
│   ├── medical_cost.py               # Medical Cost dataset loader
│   ├── income.py                     # ACS Income (Income-L / Income-A) loader
│   └── toy.py                        # Toy dataset for Figure 2 reproduction
├── data/                             # Place raw dataset files here
│   ├── Medical_Cost.zip
│   └── ACSIncome_state_number.arff
├── results/                          # Saved metrics (.json / .csv) and figures
├── report/                           # Final report PDF
├── main.py                           # ← UPDATED: AIA-only entry point (no --mode flag)
├── fl_client.py                      # ← UPDATED: FedAvg client with checkpoint hooks
├── fl_server.py                      # ← UPDATED: FedAvgWithAIA strategy
├── model_based_aia.py                # ← NEW: top-level copy with evaluate_attack_mse()
├── models.py                         # Regression models (linear / neural network)
├── utils.py                          # ← UPDATED: CSV saving, multi-seed, plotting
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Python**: 3.10 recommended (3.9+ required)  
**Seeds**: `random.seed(42)`, `numpy.seed(42)`, `torch.manual_seed(42)` — fixed globally

---

## Run Commands

> **Key change from the original**: the `--mode` flag has been removed. The entry
> point now runs AIA experiments directly. There is no longer a separate baseline mode.

### 1. Medical Cost — Passive Attack (infer smoking status, 2 clients)

```bash
python main.py \
    --dataset medical_cost \
    --data_path data/Medical_Cost.zip \
    --num_rounds 100 \
    --attack passive
```

### 2. ACS Income-L — Active Attack (infer gender, Louisiana, 10 clients)

```bash
python main.py \
    --dataset income_L \
    --data_path data/ACSIncome_state_number.arff \
    --num_rounds 100 \
    --num_clients 10 \
    --heterogeneity 0.4 \
    --attack active \
    --active_rounds 50
```

### 3. ACS Income-A — Active Attack (all 51 states)

```bash
python main.py \
    --dataset income_A \
    --data_path data/ACSIncome_state_number.arff \
    --num_rounds 100 \
    --attack active \
    --active_rounds 50
```

### 4. Gradient-Based Baseline Attack

```bash
python main.py \
    --dataset medical_cost \
    --data_path data/Medical_Cost.zip \
    --attack gradient \
    --grad_aia_iters 5000 \
    --max_grad_rounds 10
```

### 5. Run All Attack Variants on One Dataset

```bash
python main.py \
    --dataset income_L \
    --data_path data/ACSIncome_state_number.arff \
    --attack passive   # then rerun with active / gradient / none
```

### 6. Reproduce Paper Table 1 (all 3 datasets, neural network)

```bash
python main.py --reproduce_table1 --data_dir data/
```

### 7. Multi-Seed Run (mean ± std reporting)

```bash
python main.py \
    --dataset medical_cost \
    --data_path data/Medical_Cost.zip \
    --attack passive \
    --seeds 42 43 44
```

### 8. Linear Model Variant

```bash
python main.py \
    --dataset medical_cost \
    --data_path data/Medical_Cost.zip \
    --model_type linear \
    --attack passive
```

### 9. GPU Run

```bash
python main.py \
    --dataset income_A \
    --data_path data/ACSIncome_state_number.arff \
    --attack active \
    --device cuda
```

---

## Full CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `medical_cost` | `medical_cost`, `income_L`, `income_A` |
| `--data_path` | `None` | Path to dataset file (`.zip` or `.arff`) |
| `--data_dir` | `data` | Directory for `--reproduce_table1` |
| `--state_code` | `22` | FIPS code for Income-L (22 = Louisiana) |
| `--heterogeneity` | `0.4` | Heterogeneity h ∈ [0, 0.5] for Income-L split |
| `--sample_frac` | `0.2` | Per-client sample fraction for Income-A |
| `--num_rounds` | `100` | FL communication rounds |
| `--num_clients` | `None` | Override dataset default client count |
| `--local_epochs` | `1` | Local SGD epochs per round (E) |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `None` | SGD learning rate η (dataset default if None) |
| `--test_frac` | `0.1` | Fraction of data held out for testing |
| `--scale_target` | `True` | Normalise regression target |
| `--model_type` | `neural_network` | `neural_network` or `linear` |
| `--hidden_size` | `128` | Hidden layer size for neural network |
| `--attack` | `passive` | `passive`, `active`, `gradient`, `none` |
| `--targeted_client` | `0` | Index of the client to attack |
| `--active_start_round` | `None` | Round to begin active attack (default: after passive phase) |
| `--active_rounds` | `50` | Number of active attack rounds |
| `--adam_lr` | `1.0` | Adam learning rate for active adversary (Alg. 3) |
| `--adam_beta1` | `0.9` | Adam β₁ |
| `--adam_beta2` | `0.999` | Adam β₂ |
| `--grad_aia_iters` | `5000` | Gradient-based AIA optimisation iterations |
| `--max_grad_rounds` | `10` | Max rounds used for gradient AIA |
| `--checkpoint_dir` | `checkpoints` | Directory for model checkpoints |
| `--results_dir` | `results` | Directory for JSON/CSV/figure outputs |
| `--log_level` | `INFO` | Logging verbosity |
| `--seed` | `42` | Single random seed |
| `--seeds` | `None` | Multiple seeds, e.g. `--seeds 42 43 44` (reports mean ± std) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--reproduce_table1` | flag | Run all 3 datasets and produce Table 1 |

---

## What Changed vs. Original

| Area | Original | Updated |
|------|----------|---------|
| Entry point | `--mode baseline` / `--mode aia` | AIA-only — no `--mode` flag |
| Dataset choices | MNIST, FMNIST, CIFAR-10/100 + regression | Regression datasets only (`medical_cost`, `income_L`, `income_A`) |
| Attack choices | `passive`, `active` | `passive`, `active`, `gradient`, `none` |
| Multi-seed | Not supported | `--seeds 42 43 44` → mean ± std CSV |
| `model_based_aia.py` | Only in `attacks/` sub-package | Also promoted to top-level with `evaluate_attack_mse()` added |
| `utils.py` | Basic helpers | + `save_results_csv`, `save_multi_seed_csv`, `plot_accuracy_loss_curves`, `plot_aia_comparison` |
| `fl_client.py` | Minimal checkpoint stub | Full checkpoint save/load with round-indexed metadata |
| `fl_server.py` | Basic FedAvg strategy | Refined active attack injection and eavesdrop logging |
| **`aia_accuracy_percent` (bug fix)** | **Multiplied ALL float values ×100, inflating `ours_mse` (e.g. 3.22 → 321.96)** | **Only accuracy-fraction keys (`ours`, `global_model`, `grad_passive`, `final_accuracy`) scaled ×100; MSE/loss/cost stored at true scale** |
| **`table1_reproduction` output** | **JSON only** | **JSON + CSV (`table1_reproduction.csv`) — one flat row per dataset × attack variant** |
| **`fl_client.py` evaluate metric key** | **Sent MSE under `"accuracy"` key; server interpreted raw MSE as a classification accuracy fraction** | **Now sent under `"mse_loss"` key; server reads the correct key** |
| **`fl_server.py` convergence threshold** | **`avg_acc >= 0.80` — fired at round 1 for medical_cost (MSE 0.79) or never for income_L (MSE > 1)** | **Loss-decrease threshold: first round where MSE ≤ 95% of round-1 MSE** |
| **`fl_server.py` communication cost** | **`n_clients_est = 1` — always counted only 1 client regardless of federation size** | **`n_clients_est = self.min_fit_clients` — uses actual participating client count** |
| **`fl_server.py` active attacker init** | **Initialised at `active_start_round - 1`; skipped silently when `active_start_round == 1`** | **Also initialises at `active_start_round` itself as a fallback for the edge case** |
| **`main.py` CSV `final_accuracy_pct`** | **`final_mse × 100` written as a percentage (e.g. 82053.38%)** | **Renamed to `final_mse` and stored at its raw value** |

---

## Output Files

After each run, the following are written to `results/`:

| File | Description |
|------|-------------|
| `{dataset}_{attack}_r{R}_seed{S}.json` | Full result dict for a single run |
| `{dataset}_{attack}_r{R}_seed{S}.csv` | Flat CSV row for a single run |
| `{dataset}_{attack}_multiseed.csv` | Mean ± std across seeds (with `--seeds`) |
| `table1_reproduction.json` | Full Table 1 results dict (all datasets × attacks) |
| `table1_reproduction.csv` | **Table 1 results as a flat CSV** (one row per dataset × attack variant) |
| `figures/accuracy_{dataset}.png` | Global test loss vs rounds |
| `figures/aia_accuracy.png` | AIA accuracy comparison bar chart |

Checkpoints are saved to `checkpoints/{client_id}/global_{round}.pt` and `local_{round}.pt`.

---

## Key Implementation Details

- **FedAvg local update**: Algorithm 4 — SGD with `momentum=0.9`, E local epochs per round
- **Passive AIA**: collects `(θ_global, θ_local)` pairs across rounds → Linear Model Reconstruction (Algorithm 2) → Model-Based AIA (Eq. 3)
- **Active AIA**: server injects adversarial models via Adam emulation (Algorithm 3) to drive client toward local optimum
- **Gradient baseline**: cosine-similarity attribute matching (Lyu & Chen 2021), controlled by `--grad_aia_iters` and `--max_grad_rounds`
- **Proposition 1**: AIA accuracy ≥ 1 − 4E_c / θ[s]²; lower local MSE → higher attack accuracy
- **Theorem 1**: passive reconstruction error → 0 as number of collected round pairs → ∞

---

## Toy Experiment (Figure 2 Reproduction)

```bash
# Reproduce paper Figure 2 exactly (B ∈ {64, 256, 1024}, 5 seeds each)
python run_toy_experiment.py --reproduce_figure2

# Quick smoke test (1 seed)
python run_toy_experiment.py --batch_sizes 64 256 1024 --n_seeds 1

# Custom batch sizes
python run_toy_experiment.py --batch_sizes 32 128 --n_rounds 300 --lr 5e-3
```
python main.py --reproduce_table1 --data_dir data/
python main.py --reproduce_table1 --data_dir data/ --seeds 42

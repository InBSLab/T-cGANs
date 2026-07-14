# T-cGANs Ethereum Account Classification

This repository is a cleaned, modular, and leakage-controlled implementation prepared from the uploaded experimental scripts for the manuscript **“Multi-class Ethereum account De-anonymization via Conditional Generative Adversarial and Temporal Graph Attention networks.”**

> **Reproducibility status:** the repository implements the data pipeline, cGAN, temporal graph classifier, chronological validation, metrics, ablations, robustness utilities, sample-fidelity analysis, and experiment aggregation. It does **not** contain raw XBlock data, pretrained weights, or fabricated result tables. Several third-party baselines still require their official implementations and license-compatible adapters.

## What was corrected

The original upload compiled syntactically but was not suitable for a public reproducibility repository. The main corrections are:

- Removed hard-coded Windows paths and import-time data loading.
- Replaced the 1,004-line monolithic script with importable modules and one CLI.
- Standardized the ten-class label order to `Gambling=7`, `Honeypot=8`, `Ponzi=9`.
- Added the manuscript’s 25 static features and 60-dimensional per-window dynamic sequence.
- Added train-fold-only imputation, clipping, log, min-max, and z-score transformations.
- Added cGAN feature matching, gradient penalty, convergence proxy, and explicit `[-1, 1]` scaling/inverse scaling.
- Added temporal edge aggregation, pure-cosine learnable time encoding, edge-aware attention, and hierarchical layer readout.
- Added the manuscript's 60/10/30 first-activity split with an explicit guard for its unresolved observation-time contradiction; the code will not silently evaluate future accounts with no observable history.
- Corrected neighbor-sampled training so only seed nodes contribute to loss and metrics.
- Added Macro-F1, Weighted-F1, AUC, AP, MCC, per-class F1, paired tests, fidelity, robustness, and efficiency utilities.

The complete review is in [`docs/AUDIT_REPORT_CN.md`](docs/AUDIT_REPORT_CN.md), with a row-by-row manuscript mapping in [`docs/ARTICLE_CODE_ALIGNMENT.csv`](docs/ARTICLE_CODE_ALIGNMENT.csv).

## Repository structure

```text
src/tcgans/
  features.py       cleaning, 25 static features, 60-D dynamic sequences
  cgan.py           conditional generator/discriminator and training losses
  losses.py         cross-entropy, cost-sensitive, and focal loss factory
  graph.py          temporal edge aggregation and PyG graph construction
  split.py          chronological 60/10/30 split and cutoff logic
  tgat.py           cosine time encoder, TGAT classifier, static GNN controls
  train.py          leakage-safe NeighborLoader training and evaluation
  pipeline.py       end-to-end preparation and artifact persistence
  cli.py            `tcgans prepare` and `tcgans train`
experiments/
  benchmarking.py, imbalanced_baselines.py, result_analysis.py,
  synthetic_fidelity.py, adversarial_robustness.py, robustness_suite.py,
  statistical_tests.py, dataset_characterization.py, efficiency.py
configs/paper_reproduction.yaml
README_CN.md
```

## Installation

Python 3.9 is recommended to match the manuscript environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

`NeighborLoader` also needs a PyTorch/PyG-compatible sampling backend in many environments. Install the matching `pyg-lib` or `torch-sparse` build for the selected PyTorch/CUDA versions. The manuscript-oriented Conda specification is provided in `environment-paper.yml`.

## Data layout

Do not commit raw transaction data unless redistribution is explicitly permitted. The default configuration expects:

```text
data/raw/ethereum_transactions.csv
data/raw/account_labels.csv
```

Minimum transaction columns:

```text
from,to,value,timestamp
```

Optional columns used when present:

```text
gasUsed,gasPrice,gasLimit,isError,fromType,toType
```

The labels file should contain `address,label`, where `label` is either a class name or an integer from 0 to 9.

## Run the full pipeline

Edit data paths in `configs/paper_reproduction.yaml`, then run:

```bash
tcgans prepare --config configs/paper_reproduction.yaml
README_CN.md

tcgans train \
  --config configs/paper_reproduction.yaml \
  --prepared-dir artifacts/prepared/full_system \
  --output-dir results/full_system/run_0
```

Preparation saves the graph, split indices, fitted preprocessors, cGAN checkpoints/history, metadata, and real/generated feature arrays for fidelity analysis. Training saves the best classifier checkpoint, metric history, test metrics, `y_true.npy`, `y_prob.npy`, and run metadata.

## Ten-run experiments and ablations

Generate auditable configuration files instead of manually changing code:

```bash
python experiments/sweep_configs.py \
  --base configs/paper_reproduction.yaml \
  --kind temporal-ablation \
  --output-dir configs/generated/temporal \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Supported sweep templates are `time-window`, `temporal-ablation`, `component-ablation`, `component-isolation`, and `augmentation-count`. The component-isolation sweep directly creates the TGAT baseline, weighted-loss TGAT, cGAN+GCN, cGAN+GAT, and full-system configurations described around Table 7. Store each run under the convention expected by `experiments/result_analysis.py`:

```text
results/<Configuration>/run_<i>_y_true.npy
results/<Configuration>/run_<i>_y_prob.npy
```

Then aggregate real outputs:

```bash
python experiments/result_analysis.py \
  --root results \
  --configs Full_System Remove_Timestamps Remove_Temporal_Encoding Remove_Both \
  --runs 10 \
  --baseline Full_System
```


## Chronological-protocol guard

The manuscript currently combines two requirements that are not jointly operational as written: accounts are assigned to later folds by **first activity time**, while every test transaction after the training cutoff is masked. A later-fold account therefore normally has no transaction at or before that cutoff. The original standalone split script masks only edges and does not detect this issue.

The default configuration uses:

```yaml
feature_observation_mode: strict_global_cutoff
allow_unobserved_evaluation: false
```

Preparation will stop with a diagnostic rather than generate misleading metrics from featureless validation/test accounts. A `full_history` mode is available only for retrospective diagnostics; it can leak future information through node features and cannot support the manuscript's deployment claim. Before reporting results, revise the validation protocol—for example by defining account-local observation horizons or fold-specific temporal snapshots—and document the chosen protocol in the paper.

## Important unresolved method ambiguity

The manuscript does not define how a cGAN-generated **feature vector** receives transaction edges or a 60-D temporal sequence. The conservative defaults are:

```yaml
synthetic_topology: isolated
synthetic_sequence_mode: zeros
```

An optional `donor_bootstrap` topology and `donor_copy` sequence mode are implemented for experimentation, but using either changes the operational method and must be disclosed in the manuscript. Do not report results from these modes as the currently described method without revising the paper.

## Baseline policy

Feature-only RF, MLP, and optional XGBoost runners are included. GCN/GAT/GraphSAGE controls are implemented in `tcgans.tgat.StaticGNN` and selectable through `model.architecture`. Cost-sensitive and focal losses are selectable through `training.loss_strategy`. TGN, DyRep, G-Mixup, EvolveGCN, ROLAND, PEAE-GNN, GraphMixup, ImGAGN, and GraphSMOTE are intentionally marked **adapter required**. Their implementations, weights, data interfaces, licenses, and chronological protocols must be verified; placeholder outputs must never be used as experimental evidence.

## Tests

```bash
pytest
```

The included smoke tests cover feature dimensions, transformations, chronological splitting, the protocol-contradiction guard, edge aggregation, cGAN training/generation, loss strategies, sweep generation, and synthetic-fidelity metrics. Full model training was not executed in the audit environment because the raw dataset and a PyG neighbor-sampling backend were not supplied.

## Public-release checklist

Before switching the GitHub repository to public:

1. Add a copyright-holder-approved `LICENSE` file; see `LICENSE-CHOICE.md`.
2. Confirm XBlock dataset redistribution and citation terms; normally publish download/preprocessing instructions rather than raw data.
3. Add real ten-run outputs, exact random seeds, and model checksums.
4. Resolve the manuscript discrepancies and unresolved placeholders listed in the audit report.
5. Add official baseline adapters and preserve each dependency’s original license/attribution.
6. Review [`docs/DATA_AND_ETHICS.md`](docs/DATA_AND_ETHICS.md) and avoid publishing unnecessary address-level predictions.

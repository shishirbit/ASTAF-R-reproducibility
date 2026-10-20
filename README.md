# ASTAF-R reproducibility package

This repository contains the model, preprocessing, training, validation,
evaluation, baseline, sensitivity-analysis, statistical-analysis, and figure/
table generation code used for the ASTAF-R study.

## 1. Environment

Python 3.12 was used for the reported runs. Create an isolated environment and
install the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reported GPU runs used PyTorch 2.11.0 with CUDA 12.8. A CPU installation is
supported for smoke tests but is substantially slower.

## 2. Public benchmark data

METR-LA and PeMS-BAY are public traffic-speed benchmarks distributed by the
[official DCRNN repository](https://github.com/liyaguang/DCRNN). Place the four
experiment artifacts below in `data/`:

| File | Shape/use | SHA-256 |
|---|---|---|
| `metr-la.h5` | 34,272 x 207 speed series with timestamps | `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05` |
| `adj_mx_METR-LA.pkl` | DCRNN-compatible METR-LA distance graph | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |
| `pems_node_values.npy` | 52,105 x 325 x 2 speed/calendar array | `6e1503b6d76a46dd6f282468d76c14cdda0cb99073ba9edb77897594895b8eb9` |
| `pems_adj_mat.npy` | 325 x 325 PeMS-BAY graph | `bfe332913480221f0c1e67ff6c7a5ea8896a70a8fc2f6ff04b052842b91072b9` |

The release archive associated with the paper must contain the two NumPy
artifacts or provide their permanent download URL. Run `python verify_data.py`
before training. No dataset-wide statistic is fitted: `revision_data.py`
partitions the raw timeline first, estimates normalization from valid training
observations only, and builds stride-one windows independently inside each
split.

## 3. Exact protocol

- Input/output: 12 five-minute observations to 12 forecast steps.
- Split: raw-timeline chronological 70/10/20.
- Boundary rule: no input or target observation occurs in more than one split.
- Missing convention: zero/non-finite targets are masked; unavailable inputs
  are replaced by the training mean (zero after normalization).
- Seeds: 11, 22, and 33.
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-5`.
- Selection: architecture variants are compared on validation data only.
- Test: the selected architecture is written to
  `architecture_selection.json` before locked test evaluation.

The precise machine-readable configuration is in `config/round2.json`.

## 4. Reproduce

First audit the temporal split:

```bash
python experiments/audit_split_integrity.py
```

Run the validation-locked full experiment pipeline:

```bash
python experiments/run_round2_pipeline.py \
  --data-dir data \
  --output-dir results/round2_leakage_free \
  --datasets METR-LA PeMS-BAY \
  --seeds 11 22 33 \
  --batch-size 64 \
  --selection-epochs 40 --selection-patience 8 \
  --test-epochs 60 --test-patience 10
```

Completed JSON/NPZ records are reused automatically when the pipeline is
restarted. Each JSON record stores the seed, hyperparameters, environment,
dataset boundaries, early-stopping history, parameter count, timing, and test
metrics. NPZ files contain predictions, targets, masks, horizon errors, and
learned graph artifacts.

## 5. Source map

- `experiments/revision_data.py`: loading, missing-value handling,
  training-only normalization, raw-timeline splitting, calendar features, and
  data loaders.
- `experiments/revision_models.py`: ASTAF-R and every common-code baseline.
- `experiments/run_improvement_screen.py`: validation-only design studies.
- `experiments/run_revision_experiments.py`: training, early stopping,
  deterministic settings, evaluation, profiling, and artifact capture.
- `experiments/run_round2_pipeline.py`: selection lock and full experiment
  orchestration.
- `experiments/audit_split_integrity.py`: explicit intersection audit for raw
  observations used by each split.
- `results/` (added with the final versioned release): machine-readable run
  records, seed-aware statistics, graph diagnostics, tables, and plots.

The DCRNN-, STGCN-, Graph WaveNet-, AGCRN-, STID-, and Transformer-style models
in `revision_models.py` are compact common-code controls, not the authors'
official implementations. The manuscript reports them under that exact label
and separately contextualizes results from official/literature sources.

## 6. Hardware and runtime

The final run records contain the exact GPU, CUDA, PyTorch, Python, NumPy, and
SciPy versions. Runtime depends strongly on hardware. The full queue includes
multi-seed selection, diagnostic controls, two-dataset locked evaluation, and
profiling; allow at least one to two days on a single laptop GPU.

## 7. License and citation

Code is released under the MIT License; benchmark data remain under their
original providers' terms. Cite the article and archived software DOI once
those records are public.


# Z24 vibration-anomaly validation module

This module validates the **vibration-anomaly detection logic** used by SpanTwin on the Z24 Bridge benchmark. It does **not** claim that Z24 is a distributed fiber-optic sensing dataset.

## What this experiment tests

> Can vibration-only features distinguish healthy Z24 bridge records from progressive changed/damaged-condition records on an unseen measurement setup?

### Inputs intentionally excluded

- temperature
- traffic flow
- vehicle speed
- vehicle mass
- any environmental compensation variable

The experiment uses acceleration/vibration time series only.

## Scientific boundary

| SpanTwin component | Validation status |
|---|---|
| Vibration anomaly logic | Tested with Z24 acceleration signals |
| Distributed-fiber spatial visualization | Simulation / future DAS or DFOS data required |
| Fiber-specific localization performance | Not validated by Z24 |
| Daily report workflow | Simulation or uploaded fiber data required |

The correct project wording is:

> “Z24 validates the vibration-anomaly methodology. The distributed-fiber visualization and fiber-zone localization require corresponding DAS/DFOS data for direct validation.”

## Dataset access

The official Z24 benchmark is provided by KU Leuven for non-commercial research after registration. Follow the owner’s terms and acknowledge the source in project documentation.

Expected local layout after authorized acquisition:

```text
z24-validation/
├── data/
│   ├── inputs.npy             # processed windows, expected (N, channels, samples)
│   └── labels.npy             # optional scenario labels, expected (N,)
├── outputs/
├── src/
└── run_validation.py
```

A processed public derivative has been reported with `inputs.npy` shape `(1530, 27, 6000)`: 17 scenarios × 9 setups × 10 windows. The pipeline can derive ordered scenario labels only when this layout is confirmed; otherwise pass a verified `labels.npy` file.

## Reproducible experiment design

### 1. Vibration-only preprocessing

1. Convert to floating-point arrays.
2. Remove each channel’s DC component.
3. Apply a configurable bandpass filter.
4. Extract signal and frequency features.
5. Aggregate features across a selected sensor subset.

### 2. Leakage-resistant split

The benchmark windows are grouped by scenario and measurement setup. A random split would place highly similar windows from the same setup in both training and test data.

This scaffold therefore uses a **setup-based split**:

- Train: healthy scenario, setups 0–5
- Test healthy: healthy scenario, setups 6–8
- Test changed/damaged: non-healthy scenarios, setups 6–8

This is a first reproducible baseline. Exact healthy/damage scenario definitions must be verified against the official Z24 metadata before publishing results.

### 3. Baseline model

- Fit a robust `IsolationForest` only on healthy training features.
- Score unseen healthy and changed-condition windows.
- Produce ROC-AUC, average precision, thresholded false-positive rate, and a score plot.

This is intentionally interpretable. A PCA reconstruction-residual comparison can be run with `--method pca_residual`. Deep raw-signal models can be added only after these baselines are reviewed.

## Running the experiment

```bash
cd z24-validation
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_validation.py --method isolation_forest --data data/inputs.npy --labels data/labels.npy
```

If no labels file is available but the confirmed processed layout is exactly `17 × 9 × 10`, use:

```bash
python run_validation.py --data data/inputs.npy --derive-ordered-labels
```

## Independent held-out validation

```bash
python independent_validation.py --data data/inputs.npy --labels data/labels.npy
```

This uses setups 0–4 for the healthy reference baseline, setups 5–6 for channel selection, and setups 7–8 for a final untouched evaluation. It is the correct result to emphasize over exploratory channel-sweep scores.

## Exploratory processed-channel sensitivity

```bash
python channel_sweep.py --data data/inputs.npy --labels data/labels.npy
```

This evaluates one processed acceleration stream at a time with the same setup-held-out baseline. It is exploratory only: do not select a final physical sensor or claim localization until official channel metadata and an independent validation split are available.

## Outputs

```text
outputs/
├── validation_summary.json
├── window_scores.csv
└── anomaly_score_distribution.png
```

Do not report results until the source, channel selection, scenario mapping, and outputs have been reviewed.

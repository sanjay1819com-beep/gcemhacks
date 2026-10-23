#!/usr/bin/env python3
"""Independent vibration-only Z24 validation with separated channel selection.

Protocol:
  - Train reference baseline: setups 0–4
  - Channel selection: setups 5–6
  - Locked final evaluation: setups 7–8

The final test setups are not used to choose a channel. This is a preliminary
benchmark protocol because processed channel semantics and scenario labels still
require official Z24 metadata verification.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


def load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("z24_validation_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent processed-Z24 validation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/independent-validation"))
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    return parser.parse_args()


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    alerts = score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "reference_false_positive_rate": float(alerts[y == 0].mean()),
        "non_reference_alert_rate": float(alerts[y == 1].mean()),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers(Path(__file__).with_name("run_validation.py"))

    data = np.load(args.data, mmap_mode="r")
    labels = np.load(args.labels)
    if data.ndim != 3 or labels.shape != (data.shape[0],):
        raise ValueError("Expected data (N, channels, samples) and labels (N,)")

    groups = helpers.setup_ids(data.shape[0])
    reference = labels == 0
    train_mask = reference & np.isin(groups, [0, 1, 2, 3, 4])
    validation_mask = np.isin(groups, [5, 6])
    final_mask = np.isin(groups, [7, 8])
    y_validation = (~reference[validation_mask]).astype(int)
    y_final = (~reference[final_mask]).astype(int)

    if train_mask.sum() < 10 or len(np.unique(y_validation)) != 2 or len(np.unique(y_final)) != 2:
        raise RuntimeError("Unexpected split composition; verify processed dataset layout and labels")

    selection_rows = []
    channel_artifacts: dict[int, tuple[RobustScaler, IsolationForest]] = {}
    for channel in range(data.shape[1]):
        print(f"Selection sweep: processed channel {channel + 1}/{data.shape[1]}")
        features = helpers.extract_features(data, [channel], args.sampling_rate, 0.5, 30.0)
        scaler = RobustScaler().fit(features[train_mask])
        x_train = scaler.transform(features[train_mask])
        x_validation = scaler.transform(features[validation_mask])
        model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1).fit(x_train)
        train_score = -model.score_samples(x_train)
        validation_score = -model.score_samples(x_validation)
        row = {
            "channel_index": channel,
            "selection_threshold": float(np.quantile(train_score, 0.95)),
            **metrics(y_validation, validation_score, float(np.quantile(train_score, 0.95))),
        }
        selection_rows.append(row)
        channel_artifacts[channel] = (scaler, model)

    ranking = pd.DataFrame(selection_rows).sort_values("roc_auc", ascending=False)
    selected_channel = int(ranking.iloc[0]["channel_index"])
    ranking.to_csv(args.output / "validation_channel_ranking.csv", index=False)

    # Re-extract the selected channel then perform a single final untouched test.
    selected_features = helpers.extract_features(data, [selected_channel], args.sampling_rate, 0.5, 30.0)
    scaler, model = channel_artifacts[selected_channel]
    x_train = scaler.transform(selected_features[train_mask])
    x_final = scaler.transform(selected_features[final_mask])
    train_score = -model.score_samples(x_train)
    final_score = -model.score_samples(x_final)
    threshold = float(np.quantile(train_score, 0.95))
    final_metrics = metrics(y_final, final_score, threshold)

    final_frame = pd.DataFrame(
        {
            "window_index": np.flatnonzero(final_mask),
            "setup": groups[final_mask],
            "scenario_label": labels[final_mask],
            "condition": np.where(y_final == 0, "reference_scenario", "non_reference_scenario"),
            "anomaly_score": final_score,
            "alert": final_score >= threshold,
        }
    )
    final_frame.to_csv(args.output / "final_window_scores.csv", index=False)

    summary = {
        "experiment": "independent_vibration_only_processed_z24_validation",
        "protocol": {
            "training_reference_setups": [0, 1, 2, 3, 4],
            "channel_selection_setups": [5, 6],
            "final_untouched_test_setups": [7, 8],
        },
        "label_interpretation": (
            "Label 0 is provisionally treated as reference and all other labels as non-reference. "
            "Official Z24 scenario metadata is required before naming non-reference labels as damage."
        ),
        "selected_processed_channel": selected_channel,
        "selection_roc_auc": float(ranking.iloc[0]["roc_auc"]),
        "final_metrics": final_metrics,
        "final_threshold": threshold,
        "limitation": (
            "The selected index is not a verified physical bridge location. This validates a preliminary "
            "accelerometer vibration-anomaly workflow, not distributed-fiber sensing or fiber localization."
        ),
    }
    (args.output / "independent_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4.5))
    plt.hist(final_score[y_final == 0], bins=16, alpha=0.75, label="untouched reference", color="#65c9c4")
    plt.hist(final_score[y_final == 1], bins=22, alpha=0.72, label="untouched non-reference", color="#dba864")
    plt.axvline(threshold, color="#d96d75", linestyle="--", label="95% training threshold")
    plt.xlabel("IsolationForest anomaly score (higher = more anomalous)")
    plt.ylabel("Windows")
    plt.title("Independent Z24 vibration-only validation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output / "independent_score_distribution.png", dpi=170)

    print("\nLocked channel selection and untouched final test complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

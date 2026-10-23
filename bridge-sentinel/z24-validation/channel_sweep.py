#!/usr/bin/env python3
"""Exploratory vibration-only channel sensitivity sweep for a processed Z24 derivative.

The output ranks processed acceleration channels by a setup-held-out reference vs
non-reference baseline. It is exploratory: without official channel metadata and
verified scenario labels, it must not be called a physical sensor-placement or
final damage-localization result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


def load_validation_module(path: Path):
    spec = importlib.util.spec_from_file_location("z24_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z24 vibration-only channel sweep")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/channel-sweep"))
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    helpers = load_validation_module(Path(__file__).with_name("run_validation.py"))

    data = np.load(args.data, mmap_mode="r")
    labels = np.load(args.labels)
    if data.ndim != 3 or labels.shape != (data.shape[0],):
        raise ValueError("Expected data (N, channels, samples) and labels (N,)")
    groups = helpers.setup_ids(data.shape[0])
    reference = labels == 0
    train_mask = reference & np.isin(groups, [0, 1, 2, 3, 4, 5])
    test_mask = np.isin(groups, [6, 7, 8])
    y_test = (~reference[test_mask]).astype(int)

    records: list[dict] = []
    for channel in range(data.shape[1]):
        print(f"Testing processed channel {channel + 1}/{data.shape[1]}")
        features = helpers.extract_features(data, [channel], args.sampling_rate, 0.5, 30.0)
        scaler = RobustScaler().fit(features[train_mask])
        x_train = scaler.transform(features[train_mask])
        x_test = scaler.transform(features[test_mask])
        model = IsolationForest(n_estimators=250, random_state=42, n_jobs=-1).fit(x_train)
        train_score = -model.score_samples(x_train)
        score = -model.score_samples(x_test)
        threshold = float(np.quantile(train_score, 0.95))
        alert = score >= threshold
        records.append(
            {
                "channel_index": channel,
                "roc_auc": float(roc_auc_score(y_test, score)),
                "average_precision": float(average_precision_score(y_test, score)),
                "reference_false_positive_rate": float(alert[y_test == 0].mean()),
                "non_reference_alert_rate": float(alert[y_test == 1].mean()),
                "threshold": threshold,
            }
        )

    results = pd.DataFrame(records).sort_values("roc_auc", ascending=False)
    results.to_csv(args.output / "channel_sensitivity.csv", index=False)
    summary = {
        "experiment": "preliminary_processed_z24_channel_sensitivity",
        "interpretation": (
            "Exploratory ranking of processed acceleration channels under a vibration-only, "
            "setup-held-out baseline. Channel meanings and scenario labels require official metadata verification."
        ),
        "data_shape": list(map(int, data.shape)),
        "reference_label": 0,
        "top_five_channels_by_roc_auc": results.head(5).to_dict(orient="records"),
    }
    (args.output / "channel_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTop five exploratory channels")
    print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

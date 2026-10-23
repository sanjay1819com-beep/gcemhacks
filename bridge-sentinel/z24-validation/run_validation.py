#!/usr/bin/env python3
"""Vibration-only Z24 anomaly-detection baseline.

This script expects an array shaped (windows, channels, samples). It fits an
IsolationForest to healthy vibration windows from training measurement setups,
then evaluates healthy and changed-condition windows from held-out setups.

It deliberately avoids temperature, vehicle speed/mass, traffic flow and other
operational/environmental inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z24 vibration-only anomaly baseline")
    parser.add_argument("--data", type=Path, required=True, help="Path to inputs.npy")
    parser.add_argument("--labels", type=Path, help="Optional path to scenario labels.npy")
    parser.add_argument(
        "--derive-ordered-labels",
        action="store_true",
        help="Derive labels only for a confirmed 17 scenarios × 9 setups × 10 windows layout.",
    )
    parser.add_argument(
        "--channels",
        default="0,1,2,3,4,5",
        help="Comma-separated sensor channel indices. Default: first six channels.",
    )
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--low-hz", type=float, default=0.5)
    parser.add_argument("--high-hz", type=float, default=30.0)
    parser.add_argument("--healthy-label", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=["isolation_forest", "pca_residual"],
        default="isolation_forest",
        help="Vibration-only anomaly score method.",
    )
    parser.add_argument("--train-setups", default="0,1,2,3,4,5")
    parser.add_argument("--test-setups", default="6,7,8")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


def ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_labels(args: argparse.Namespace, n_windows: int) -> np.ndarray:
    if args.labels:
        labels = np.load(args.labels)
        if labels.shape != (n_windows,):
            raise ValueError(f"labels shape {labels.shape}; expected ({n_windows},)")
        return labels.astype(int)

    if args.derive_ordered_labels:
        expected = 17 * 9 * 10
        if n_windows != expected:
            raise ValueError(
                "Ordered label derivation is only permitted for a confirmed 17×9×10 layout; "
                f"received {n_windows} windows."
            )
        return np.repeat(np.arange(17), 9 * 10)

    raise ValueError("Provide --labels or use --derive-ordered-labels only after confirming the layout.")


def setup_ids(n_windows: int, scenarios: int = 17, setups: int = 9, segments: int = 10) -> np.ndarray:
    expected = scenarios * setups * segments
    if n_windows != expected:
        raise ValueError(
            "Setup IDs cannot be inferred because window count is not 17×9×10. "
            "Provide a verified grouping file before evaluating."
        )
    return np.tile(np.repeat(np.arange(setups), segments), scenarios)


def bandpass(windows: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist = fs / 2
    if not 0 < low < high < nyquist:
        raise ValueError(f"Invalid bandpass [{low}, {high}] for sampling rate {fs}")
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    demeaned = windows - windows.mean(axis=-1, keepdims=True)
    return sosfiltfilt(sos, demeaned, axis=-1)


def channel_features(signal: np.ndarray, fs: float) -> list[float]:
    """Return vibration-only time/frequency features for a single channel."""
    rms = float(np.sqrt(np.mean(signal**2)))
    std = float(np.std(signal))
    peak = float(np.max(np.abs(signal)))
    crest = peak / (rms + 1e-12)
    f, psd = welch(signal, fs=fs, nperseg=min(1024, signal.size))
    psd_sum = float(np.sum(psd)) + 1e-12
    centroid = float(np.sum(f * psd) / psd_sum)
    dominant = float(f[int(np.argmax(psd))])

    def band_power(lo: float, hi: float) -> float:
        mask = (f >= lo) & (f < hi)
        return float(trapezoid(psd[mask], f[mask])) if np.any(mask) else 0.0

    return [
        rms,
        std,
        peak,
        crest,
        centroid,
        dominant,
        band_power(0.5, 5.0),
        band_power(5.0, 15.0),
        band_power(15.0, 30.0),
    ]


def extract_features(data: np.ndarray, channels: list[int], fs: float, low: float, high: float) -> np.ndarray:
    """Extract features one window at a time.

    Z24 derivatives can be close to 1 GB. Streaming windows from a memory-mapped
    array avoids materializing the whole filtered data cube in RAM.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (windows, channels, samples), received {data.shape}")
    if min(channels) < 0 or max(channels) >= data.shape[1]:
        raise IndexError(f"Requested channels {channels}, data has {data.shape[1]} channels")

    feature_count = len(channels) * 9
    features = np.empty((data.shape[0], feature_count), dtype=np.float64)
    for index in range(data.shape[0]):
        # A single six-channel window is only a few hundred kilobytes.
        window = np.asarray(data[index, channels, :], dtype=np.float64)
        filtered_window = bandpass(window, fs, low, high)
        values: list[float] = []
        for channel in filtered_window:
            values.extend(channel_features(channel, fs))
        features[index] = values
        if (index + 1) % 100 == 0 or index + 1 == data.shape[0]:
            print(f"  feature extraction: {index + 1}/{data.shape[0]} windows")
    return features


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, mmap_mode="r")
    labels = load_labels(args, data.shape[0])
    groups = setup_ids(data.shape[0])
    channels = ints(args.channels)
    train_setups = ints(args.train_setups)
    test_setups = ints(args.test_setups)

    print(f"Loading {data.shape[0]} windows × {data.shape[1]} channels × {data.shape[2]} samples")
    print(f"Selected channels: {channels}")
    features = extract_features(data, channels, args.sampling_rate, args.low_hz, args.high_hz)

    healthy = labels == args.healthy_label
    train_mask = healthy & np.isin(groups, train_setups)
    test_mask = np.isin(groups, test_setups)
    if train_mask.sum() < 10 or test_mask.sum() < 10:
        raise RuntimeError("Insufficient train/test windows after setup-based split")

    scaler = RobustScaler().fit(features[train_mask])
    x_train = scaler.transform(features[train_mask])
    x_test = scaler.transform(features[test_mask])
    y_test = (~healthy[test_mask]).astype(int)

    if args.method == "isolation_forest":
        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ).fit(x_train)
        # Higher score means more anomalous for report readability.
        train_score = -model.score_samples(x_train)
        anomaly_score = -model.score_samples(x_test)
    else:
        # Reconstruction residual provides an interpretable vibration-only baseline.
        model = PCA(n_components=0.95, svd_solver="full", random_state=42).fit(x_train)
        train_score = np.mean((x_train - model.inverse_transform(model.transform(x_train))) ** 2, axis=1)
        anomaly_score = np.mean((x_test - model.inverse_transform(model.transform(x_test))) ** 2, axis=1)

    healthy_scores = anomaly_score[y_test == 0]
    threshold = float(np.quantile(train_score, 0.95))
    prediction = (anomaly_score >= threshold).astype(int)

    metrics = {
        "experiment": f"vibration_only_z24_{args.method}_preliminary",
        "method": args.method,
        "label_interpretation": (
            "Label 0 is provisionally treated as the reference class and every other label as a non-reference "
            "scenario. Official Z24 scenario metadata must be checked before calling any non-reference class damage."
        ),
        "data_shape": list(map(int, data.shape)),
        "selected_channels": channels,
        "sampling_rate_hz": args.sampling_rate,
        "bandpass_hz": [args.low_hz, args.high_hz],
        "healthy_label": args.healthy_label,
        "train_setups": train_setups,
        "test_setups": test_setups,
        "train_windows": int(train_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "test_healthy_windows": int((y_test == 0).sum()),
        "test_non_reference_windows": int((y_test == 1).sum()),
        "threshold_95pct_healthy_train": threshold,
        "roc_auc": float(roc_auc_score(y_test, anomaly_score)) if len(np.unique(y_test)) == 2 else None,
        "average_precision": float(average_precision_score(y_test, anomaly_score)) if len(np.unique(y_test)) == 2 else None,
        "test_false_positive_rate": float((prediction[y_test == 0] == 1).mean()) if healthy_scores.size else None,
        "test_non_reference_alert_rate": float((prediction[y_test == 1] == 1).mean()) if np.any(y_test == 1) else None,
        "limitation": (
            "This validates vibration anomaly detection on accelerometer records. "
            "It does not directly validate distributed-fiber sensing or fiber-specific localization."
        ),
    }

    with (args.output / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    result_frame = pd.DataFrame(
        {
            "window_index": np.flatnonzero(test_mask),
            "setup": groups[test_mask],
            "scenario_label": labels[test_mask],
            "condition": np.where(y_test == 1, "non_reference_scenario", "reference_scenario"),
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "alert": prediction,
        }
    )
    result_frame.to_csv(args.output / "window_scores.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.hist(anomaly_score[y_test == 0], bins=20, alpha=0.75, label="held-out healthy", color="#65c9c4")
    plt.hist(anomaly_score[y_test == 1], bins=20, alpha=0.72, label="non-reference scenarios", color="#dba864")
    plt.axvline(threshold, color="#d96d75", linewidth=1.5, linestyle="--", label="95% train threshold")
    plt.xlabel(f"{args.method} anomaly score (higher = more anomalous)")
    plt.ylabel("Windows")
    plt.title("Z24 vibration-only anomaly score distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output / "anomaly_score_distribution.png", dpi=170)

    print("\nValidation complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

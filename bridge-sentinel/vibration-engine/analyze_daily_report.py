#!/usr/bin/env python3
"""Vibration-only daily distributed-fiber report analysis.

The input schema is intentionally limited to vibration data. This script performs
spatial residual screening and report-to-report persistence tracking; it does not
use temperature, traffic, speed, mass, weather, or environmental compensation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "fiber_position_m", "window", "vibration_value"}
WINDOWS = ("pre_event", "passage_event", "post_event")


@dataclass
class Zone:
    name: str
    start_m: float
    end_m: float
    virtual_positions: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpanTwin vibration-only daily report engine")
    parser.add_argument("--input", type=Path, required=True, help="Daily vibration CSV")
    parser.add_argument("--history", type=Path, help="Optional previous summaries JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output report JSON")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Robust residual threshold in median absolute deviations (default: 3.0)",
    )
    parser.add_argument(
        "--minimum-cluster-points",
        type=int,
        default=3,
        help="Minimum contiguous fiber positions to form a probable zone (default: 3)",
    )
    return parser.parse_args()


def load_report(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["fiber_position_m"] = pd.to_numeric(frame["fiber_position_m"], errors="coerce")
    frame["vibration_value"] = pd.to_numeric(frame["vibration_value"], errors="coerce")
    frame["window"] = frame["window"].astype(str).str.strip().str.lower()
    frame = frame.dropna(subset=["fiber_position_m", "vibration_value"])
    frame = frame[frame["window"].isin(WINDOWS)]
    if frame.empty:
        raise ValueError("No valid vibration rows remain after validation")
    return frame


def contiguous_groups(indices: np.ndarray, positions: np.ndarray) -> list[np.ndarray]:
    """Group flagged points only when they are adjacent along the fiber route.

    Array index adjacency alone is unsafe when a cable section has missing
    virtual positions. A cluster must also respect the observed median spatial
    sampling interval, so a result cannot bridge an unobserved cable gap.
    """
    if indices.size == 0:
        return []
    all_gaps = np.diff(positions)
    median_spacing = float(np.median(all_gaps)) if all_gaps.size else 0.0
    maximum_contiguous_gap = max(median_spacing * 1.5, 1e-12)
    flagged_positions = positions[indices]
    breaks = np.where(
        (np.diff(indices) > 1) | (np.diff(flagged_positions) > maximum_contiguous_gap)
    )[0] + 1
    return list(np.split(indices, breaks))


def map_zone(positions: np.ndarray, bridge_min: float, bridge_max: float) -> str:
    center = float(np.mean(positions))
    relative = (center - bridge_min) / max(bridge_max - bridge_min, 1e-12)
    if relative < 1 / 3:
        return "West deck-girder"
    if relative < 2 / 3:
        return "Midspan deck-girder"
    return "East deck-girder"


def load_history(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("History file must contain a JSON list of prior summaries")
    return data


def calculate_persistence(history: list[dict[str, Any]], zone_name: str | None, current_date: str) -> int:
    """Count consecutive prior report summaries for the same probable zone.

    History is sorted by reported date rather than upload order, so backfilling a
    daily report does not silently change persistence logic.
    """
    if not zone_name:
        return 0

    def report_date(report: dict[str, Any]) -> str:
        dates = report.get("report_dates", [])
        return max(dates) if isinstance(dates, list) and dates else ""

    prior = [report for report in history if report_date(report) <= current_date]
    prior.sort(key=report_date, reverse=True)
    count = 1  # current report
    for report in prior:
        if report.get("probable_zone", {}).get("name") == zone_name:
            count += 1
        else:
            break
    return count


def analyze(frame: pd.DataFrame, history: list[dict[str, Any]], z_threshold: float, minimum_cluster_points: int) -> dict[str, Any]:
    dates = sorted(frame["date"].astype(str).unique())
    pivot = frame.pivot_table(
        index="fiber_position_m",
        columns="window",
        values="vibration_value",
        aggfunc="mean",
    ).reindex(columns=WINDOWS)
    complete = pivot.dropna().sort_index()
    if len(complete) < minimum_cluster_points:
        raise ValueError(
            f"Only {len(complete)} complete fiber positions have all three windows; "
            f"need at least {minimum_cluster_points}."
        )

    baseline = complete["pre_event"].to_numpy()
    passage = complete["passage_event"].to_numpy()
    recovery = complete["post_event"].to_numpy()
    positions = complete.index.to_numpy(dtype=float)

    # Normalized vibration-only change.  The recovery term helps identify a
    # change that remains after the passage event rather than temporary response.
    denominator = np.abs(baseline) + np.median(np.abs(baseline)) * 0.05 + 1e-12
    passage_change = np.abs(passage - baseline) / denominator
    recovery_change = np.abs(recovery - baseline) / denominator
    residual = 0.60 * passage_change + 0.40 * recovery_change

    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median))) + 1e-12
    robust_z = 0.6745 * (residual - median) / mad
    candidate_indices = np.flatnonzero(robust_z >= z_threshold)
    groups = [
        group
        for group in contiguous_groups(candidate_indices, positions)
        if len(group) >= minimum_cluster_points
    ]

    bridge_min, bridge_max = float(positions.min()), float(positions.max())
    probable_zone: Zone | None = None
    if groups:
        # Select the spatially coherent group with the highest aggregate residual.
        best = max(groups, key=lambda group: float(residual[group].sum()))
        affected = positions[best]
        probable_zone = Zone(
            name=map_zone(affected, bridge_min, bridge_max),
            start_m=round(float(affected.min()), 2),
            end_m=round(float(affected.max()), 2),
            virtual_positions=[round(float(value), 2) for value in affected],
        )

    persistence_days = calculate_persistence(history, probable_zone.name if probable_zone else None, max(dates))
    position_gaps = np.diff(positions)
    median_spacing = float(np.median(position_gaps)) if position_gaps.size else 0.0
    maximum_gap = float(np.max(position_gaps)) if position_gaps.size else 0.0
    quality = {
        "valid_rows": int(len(frame)),
        "complete_virtual_positions": int(len(complete)),
        "missing_position_fraction": round(1 - len(complete) / max(frame["fiber_position_m"].nunique(), 1), 4),
        "median_fiber_position_spacing_m": round(median_spacing, 3),
        "maximum_fiber_position_gap_m": round(maximum_gap, 3),
        "spatial_continuity_rule": "Flagged positions must be no more than 1.5 times the observed median spacing apart.",
    }
    if probable_zone is None:
        state = "normal"
        recommendation = "No spatially coherent vibration-deviation zone identified in this report."
    elif persistence_days >= 3:
        state = "persistent_zone"
        recommendation = "Persistent zone: targeted visual inspection is recommended."
    else:
        state = "observe_next_report"
        recommendation = "Local deviation found: compare the same zone in the next daily vibration report."

    report = {
        "analysis_type": "vibration_only_daily_spatial_screening",
        "report_dates": dates,
        "state": state,
        "recommendation": recommendation,
        "probable_zone": asdict(probable_zone) if probable_zone else None,
        "persistence_days": persistence_days,
        "data_quality": quality,
        "window_means": {
            "pre_event": round(float(np.mean(baseline)), 6),
            "passage_event": round(float(np.mean(passage)), 6),
            "post_event": round(float(np.mean(recovery)), 6),
        },
        "residual_summary": {
            "median": round(median, 6),
            "maximum": round(float(np.max(residual)), 6),
            "threshold_robust_z": z_threshold,
            "candidate_positions": int(len(candidate_indices)),
            "coherent_clusters": int(len(groups)),
        },
        "position_results": [
            {
                "fiber_position_m": round(float(pos), 3),
                "pre_event": round(float(pre), 6),
                "passage_event": round(float(event), 6),
                "post_event": round(float(after), 6),
                "vibration_residual": round(float(res), 6),
                "robust_z": round(float(z), 6),
            }
            for pos, pre, event, after, res, z in zip(positions, baseline, passage, recovery, residual, robust_z)
        ],
        "limitation": (
            "A probable zone is a vibration-screening result for targeted inspection. "
            "It is not an exact crack location or proof of structural failure."
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    frame = load_report(args.input)
    history = load_history(args.history)
    report = analyze(frame, history, args.z_threshold, args.minimum_cluster_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("state", "probable_zone", "persistence_days", "recommendation")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repeatable smoke tests for the SpanTwin vibration-only software prototype."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "vibration-engine" / "analyze_daily_report.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("vibration_engine_test", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load vibration engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_report(path: Path, report_date: date, east_deviation: bool) -> None:
    rows: list[list[str]] = []
    for index, position in enumerate(np.linspace(0, 100, 128)):
        baseline = 0.180 + 0.002 * np.sin(index * 0.31)
        passage = 0.292 + 0.003 * np.cos(index * 0.19)
        recovery = 0.184 + 0.002 * np.sin(index * 0.27)
        if east_deviation and 72 <= position <= 88:
            passage += 0.18 + 0.012 * np.sin(index)
            recovery += 0.12 + 0.009 * np.cos(index)
        rows.extend(
            [
                [report_date.isoformat(), f"{position:.3f}", "pre_event", f"{baseline:.6f}"],
                [report_date.isoformat(), f"{position:.3f}", "passage_event", f"{passage:.6f}"],
                [report_date.isoformat(), f"{position:.3f}", "post_event", f"{recovery:.6f}"],
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "fiber_position_m", "window", "vibration_value"])
        writer.writerows(rows)


def test_engine_persistence() -> None:
    engine = load_engine()
    summaries: list[dict] = []
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        start = date(2026, 10, 22)
        for day in range(3):
            csv_path = temp / f"report_{day}.csv"
            make_report(csv_path, start + timedelta(days=day), east_deviation=True)
            frame = engine.load_report(csv_path)
            result = engine.analyze(frame, summaries, z_threshold=3.0, minimum_cluster_points=3)
            summaries.append(
                {
                    "report_dates": result["report_dates"],
                    "probable_zone": result["probable_zone"],
                }
            )

    assert result["state"] == "persistent_zone", result
    assert result["probable_zone"]["name"] == "East deck-girder", result
    assert result["persistence_days"] == 3, result
    assert 72 <= result["probable_zone"]["start_m"] <= 73, result
    assert 87 <= result["probable_zone"]["end_m"] <= 88, result
    assert len(result["position_results"]) == 128, result


def test_engine_normal_case() -> None:
    engine = load_engine()
    with tempfile.TemporaryDirectory() as temporary:
        csv_path = Path(temporary) / "normal.csv"
        make_report(csv_path, date(2026, 10, 24), east_deviation=False)
        result = engine.analyze(engine.load_report(csv_path), [], z_threshold=3.0, minimum_cluster_points=3)
    assert result["state"] == "normal", result
    assert result["probable_zone"] is None, result
    assert result["persistence_days"] == 0, result


def test_spatial_gap_cannot_become_one_zone() -> None:
    """Three flagged values separated by long fiber gaps must not form a cluster."""
    engine = load_engine()
    rows: list[list[str]] = []
    positions = [float(value) for value in range(20)] + [100.0, 200.0, 300.0]
    for position in positions:
        anomalous = position >= 100.0
        values = {
            "pre_event": 0.20,
            "passage_event": 0.70 if anomalous else 0.25,
            "post_event": 0.50 if anomalous else 0.22,
        }
        rows.extend([["2026-10-24", f"{position:.2f}", window, f"{value:.3f}"] for window, value in values.items()])
    with tempfile.TemporaryDirectory() as temporary:
        csv_path = Path(temporary) / "gappy.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["date", "fiber_position_m", "window", "vibration_value"])
            writer.writerows(rows)
        result = engine.analyze(engine.load_report(csv_path), [], z_threshold=3.0, minimum_cluster_points=3)

    assert result["state"] == "normal", result
    assert result["residual_summary"]["candidate_positions"] == 3, result
    assert result["data_quality"]["maximum_fiber_position_gap_m"] == 100.0, result


def test_z24_outputs_exist() -> None:
    for directory in ("preliminary-processed-derivative", "preliminary-pca-residual"):
        summary_path = ROOT / "z24-validation" / "outputs" / directory / "validation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["data_shape"] == [1530, 27, 6000], summary
        assert "roc_auc" in summary, summary

    independent = json.loads(
        (ROOT / "z24-validation" / "outputs" / "independent-validation" / "independent_validation_summary.json").read_text(encoding="utf-8")
    )
    assert independent["selected_processed_channel"] == 4, independent
    assert "roc_auc" in independent["final_metrics"], independent


def main() -> None:
    test_engine_normal_case()
    print("PASS: normal vibration report remains normal")
    test_engine_persistence()
    print("PASS: three daily east-zone reports produce persistent-zone result")
    test_spatial_gap_cannot_become_one_zone()
    print("PASS: spatial gaps cannot be misreported as one fiber zone")
    test_z24_outputs_exist()
    print("PASS: preliminary Z24 validation outputs are present")
    print("All SpanTwin smoke tests passed.")


if __name__ == "__main__":
    main()

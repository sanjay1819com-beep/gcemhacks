#!/usr/bin/env python3
"""SpanTwin local web server and vibration-only daily-report API.

Run:
    python server.py --port 4173

The server hosts the desktop website and exposes a local API that analyzes
uploaded CSV vibration reports with vibration-engine/analyze_daily_report.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "daily-reports"
ANALYSIS_DIR = REPORT_DIR / "analyses"
HISTORY_FILE = REPORT_DIR / "history.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def load_engine():
    engine_path = ROOT / "vibration-engine" / "analyze_daily_report.py"
    spec = importlib.util.spec_from_file_location("vibration_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load vibration engine: {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = load_engine()


def history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def safe_file_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "daily_report"


class SpanTwinHandler(SimpleHTTPRequestHandler):
    """Static file server plus minimal local JSON/CSV API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def send_json(self, data: dict | list, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"status": "ok", "service": "spantwin-vibration-api"})
            return
        if path == "/api/reports":
            self.send_json(history())
            return
        if path.startswith("/api/reports/"):
            analysis_id = safe_file_stem(path.removeprefix("/api/reports/"))
            result_file = ANALYSIS_DIR / f"{analysis_id}.json"
            if not result_file.exists():
                self.send_json({"error": "Saved report not found."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(json.loads(result_file.read_text(encoding="utf-8")))
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/analyze-report":
            self.send_json({"error": "Unknown API endpoint"}, HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self.send_json(
                {"error": f"CSV upload must be between 1 byte and {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
                HTTPStatus.BAD_REQUEST,
            )
            return

        raw = self.rfile.read(length)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            self.send_json({"error": "CSV must be UTF-8 encoded."}, HTTPStatus.BAD_REQUEST)
            return

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        filename = safe_file_stem(self.headers.get("X-Report-Filename", "daily_report.csv"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        analysis_id = f"{timestamp}_{Path(filename).stem}"
        upload_path = REPORT_DIR / f"{analysis_id}.csv"
        upload_path.write_text(text, encoding="utf-8")

        try:
            frame = ENGINE.load_report(upload_path)
            result = ENGINE.analyze(frame, history(), z_threshold=3.0, minimum_cluster_points=3)
        except (ValueError, KeyError) as exc:
            upload_path.unlink(missing_ok=True)
            self.send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        except Exception as exc:  # avoid returning a stack trace to the browser
            upload_path.unlink(missing_ok=True)
            self.send_json({"error": f"Analysis failed: {type(exc).__name__}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        summary = {
            "analysis_id": analysis_id,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "source_file": filename,
            "input_mode": "distributed_fiber_vibration_csv",
            "report_dates": result["report_dates"],
            "state": result["state"],
            "probable_zone": result["probable_zone"],
            "persistence_days": result["persistence_days"],
            "recommendation": result["recommendation"],
        }
        all_history = history()
        all_history.append(summary)
        HISTORY_FILE.write_text(json.dumps(all_history, indent=2), encoding="utf-8")
        result["source"] = "distributed_fiber_vibration_csv"
        result["analysis_id"] = analysis_id
        result["stored_summary"] = summary
        (ANALYSIS_DIR / f"{analysis_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.send_json(result, HTTPStatus.CREATED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SpanTwin and its local vibration API")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), SpanTwinHandler)
    print(f"SpanTwin server running on http://0.0.0.0:{args.port}")
    print("API: GET /api/health, GET /api/reports, POST /api/analyze-report")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

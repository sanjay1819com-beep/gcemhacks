# SpanTwin — Bridge Damage Detection and Localization

A desktop-first civil-infrastructure software prototype for vibration-only bridge monitoring.

## What the prototype includes

- Cinematic cable-stayed bridge digital twin
- One visible continuous distributed-fiber sensing loop
- Scroll-driven vehicle-passage visualization
- 128 virtual sensing positions along the fiber route
- Vibration-only spatial anomaly simulations
- Daily before / passage / after vibration reports
- Persistent-zone logic across daily reports
- Local CSV report upload and Python analysis API
- Report download / print
- Preliminary Z24 vibration-anomaly validation module with explicit limitations

## Run the website and API

```bash
cd bridge-sentinel
python3 -m pip install -r requirements.txt
python3 server.py --port 4173
```

Open the live preview on port `4173`.

## Daily report CSV schema

```csv
date,fiber_position_m,window,vibration_value
2026-10-24,72.0,pre_event,0.18
2026-10-24,72.0,passage_event,0.42
2026-10-24,72.0,post_event,0.26
```

Required windows:

- `pre_event`
- `passage_event`
- `post_event`

The API creates a probable zone only when a spatially contiguous group of fiber positions has a robust vibration residual. Flagged positions must also be no more than 1.5 times the report's median fiber-position spacing apart, so a missing cable segment cannot be reported as one continuous zone. Repeated reports from the same zone are tracked as persistence.

## Local API

```text
GET  /api/health
GET  /api/reports
GET  /api/reports/{analysis_id}
POST /api/analyze-report
```

`POST /api/analyze-report` accepts raw UTF-8 CSV in the request body with content type `text/csv`. The web interface handles this automatically.

## Important interpretation

A result such as `East deck-girder / 72–88 m` means a **probable vibration-deviation zone for targeted inspection**. It is not an exact crack coordinate and does not prove structural failure.

## Z24 validation module

See `z24-validation/README.md` and `z24-validation/PRELIMINARY_RESULTS.md`.

The current Z24 work is a preliminary vibration-anomaly baseline on a processed public derivative. It validates neither distributed-fiber sensing nor fiber-specific localization. Official Z24 metadata and authorized data remain necessary for final benchmark claims.

## Test command

```bash
python3 tests/smoke_test.py
```

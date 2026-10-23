# Preliminary Z24 processed-derivative result

**Status:** preliminary reproducibility experiment — not a final damage-validation claim.

## Source status

- Input: processed public Z24 derivative used for a fast preliminary run.
- Source type: derivative, not the official KU Leuven delivery.
- The original data source must be acknowledged and official scenario metadata must be checked before publication or final claims.
- No temperature, traffic, vehicle speed, vehicle mass, or other environmental/operational variable was used as an input.

## Experiment

| Item | Choice |
|---|---|
| Data shape | 1,530 windows × 27 channels × 6,000 samples |
| Selected channels | 0–5 (provisional; sensor semantics not yet verified) |
| Sampling rate | 100 Hz |
| Bandpass | 0.5–30 Hz |
| Features | RMS, standard deviation, peak, crest factor, spectral centroid, dominant frequency, and 3 band powers per channel |
| Detector | IsolationForest trained only on reference-label windows |
| Split | Train reference windows from setups 0–5; test setups 6–8 |
| Reference interpretation | Label `0` provisionally treated as reference; labels `1–16` treated only as non-reference scenarios |

## Output

| Metric | Result | Correct interpretation |
|---|---:|---|
| ROC-AUC | 0.628 | Modest reference vs non-reference separation for this simple baseline. |
| Average precision | 0.968 | Inflated by the very imbalanced test composition: 480 non-reference vs 30 reference windows. Do not use alone. |
| Held-out reference false-positive rate | 0.000 | Based on only 30 held-out reference windows; this is not a robust field false-alarm estimate. |
| Non-reference alert rate at 95th-percentile reference threshold | 0.148 | Low sensitivity for the current baseline. It is not ready to be presented as a high-performing detector. |

## Method comparison

| Method | ROC-AUC | Held-out reference false-positive rate | Non-reference alert rate | Decision |
|---|---:|---:|---:|---|
| IsolationForest features | 0.628 | 0.000 | 0.148 | Keep as the current preliminary baseline; separation is modest. |
| PCA reconstruction residual | 0.556 | 0.500 | 0.506 | Do not use as the current detector; held-out reference false alarms are too high. |

Both comparisons use the same provisional label interpretation and the same setup-held-out split. Neither should be reported as final damage accuracy.

## Independent held-out validation

To avoid evaluating a channel on the same setups that selected it, a second protocol was run:

```text
Train reference baseline: setups 0–4
Channel selection: setups 5–6
Final untouched evaluation: setups 7–8
```

| Metric | Result |
|---|---:|
| Selected processed channel on selection setups | 4 |
| Selection ROC-AUC | 0.866 |
| Final untouched-test ROC-AUC | 0.561 |
| Final held-out reference false-positive rate | 0.000 on 20 reference windows |
| Final non-reference alert rate | 0.091 |

The large drop from selection to final test is important: this simple pipeline does **not** generalize strongly enough to be presented as a final detector. It is evidence that setup variability and unverified labels/channel semantics matter. This is a scientifically useful negative result, not a failure to hide.

Files:

- `outputs/independent-validation/independent_validation_summary.json`
- `outputs/independent-validation/validation_channel_ranking.csv`
- `outputs/independent-validation/final_window_scores.csv`
- `outputs/independent-validation/independent_score_distribution.png`

## Exploratory processed-channel sensitivity

A vibration-only single-channel sweep was also run across all 27 processed acceleration streams using the same held-out setup split.

| Processed channel index | ROC-AUC | Reference false-positive rate | Interpretation |
|---:|---:|---:|---|
| 6 | 0.778 | 0.033 | Best exploratory single stream |
| 3 | 0.760 | 0.000 | Second-best exploratory single stream |
| 20 | 0.757 | 0.000 | Third-best exploratory single stream |
| 17 | 0.734 | 0.000 | Exploratory candidate |
| 16 | 0.733 | 0.000 | Exploratory candidate |

This is **not** a final sensor-selection claim. The channel ranking used the same held-out setup data, and the processed-channel physical meanings have not been verified against official Z24 metadata. It is useful only to show that vibration sensitivity differs between recorded channels and that validated channel selection matters.

Files:

- `outputs/channel-sweep/channel_sensitivity.csv`
- `outputs/channel-sweep/channel_sensitivity_summary.json`

## What this result proves

- The end-to-end vibration-only pipeline executes reproducibly on a real processed Z24 derivative.
- The pipeline avoids random window splitting by holding out measurement setups.
- A simple detector produces a measurable but weak reference-vs-non-reference separation.

## What this result does **not** prove

- It does not prove damage detection for every non-reference label.
- It does not validate the distributed-fiber sensing concept.
- It does not validate fiber-specific localization.
- It must not be presented as a final accuracy result.

## Required next experiments

1. Verify official Z24 scenario metadata and which labels are genuinely undamaged, environmental, and damaged.
2. Verify sensor/channel meanings and use the appropriate reference channels across setups.
3. Compare at least three vibration-only methods:
   - IsolationForest features (current baseline)
   - PCA / reconstruction residual features
   - MiniRocket or another raw-signal model
4. Report results by verified damage scenario, not only a collapsed label group.
5. Re-run the same code with authorized official KU Leuven data when registration is available.

## Files produced

- `outputs/preliminary-processed-derivative/validation_summary.json`
- `outputs/preliminary-processed-derivative/window_scores.csv`
- `outputs/preliminary-processed-derivative/anomaly_score_distribution.png`

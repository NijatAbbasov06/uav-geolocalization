# Diagnostic plots

Every figure generated during the work, including the intermediate ones that are
not in the report. The report uses 15 of these; the rest document how the bugs
were found and are kept because they show *why* certain decisions were made.

## Results

| plot | what it shows |
|---|---|
| `FINAL_causal_dataset4.png` | the headline result — real-time fusion, 464 → 198 m |
| `FINAL_causal_run2.png` | the same for OpenVINS run 2 |
| `FINAL_dataset4_q200.png` | offline fusion with the altitude profile |
| `FINAL_dataset4.png` | offline fusion on the older 400 m tile database |
| `run_variance.png` | three OpenVINS runs, identical inputs |
| `run1_vs_run2.png` | two of those side by side |
| `fused_dataset4.png` | the EKF failing — scale freezes as the gate closes |
| `compute_cost.png` | cores, memory and rate for both halves |

## Retrieval

| plot | what it shows |
|---|---|
| `caevl_dataset4_map.png` | where retrieval succeeds on dataset4 |
| `caevl_munfrl_map.png` | the same for dataset3 (forest — mostly fails) |
| `caevl_munfrl_error.png` | error against altitude on dataset3 |
| `caevl_q200_compare.png` | restricting the tile database, 17.7% → 35.8% |
| `caevl_predictions_dataset4.png` | every prediction joined to its true position |
| `consensus_filter.png` | outlier rejection, 17.7% → 99.0% precision |
| `dataset3_vs_dataset4_terrain.png` | the two flights over their reference maps |
| `coverage_dataset4.png` | downloaded imagery vs the tiled corridor |
| `whole_flight_context.png` | where queries exist within the whole flight |

## VIO

| plot | what it shows |
|---|---|
| `vio_anchoring_dataset4.png` | anchored at takeoff vs best-fit placement |
| `vio_dataset4.png` | OpenVINS against truth, and the scale error |
| `vio_traj.png`, `vio_drift.png` | earlier trajectory and drift plots |

## Finding the 456 m georeferencing bug

Read in this order — it is the clearest record of how the error surfaced.

| plot | what it shows |
|---|---|
| `munfrl_pairs.png` | the symptom: query shows forest, its "matching" tile shows farmland, 6 m apart |
| `track_on_map.png` | the cause: the track projected with the shipped parameters (red) against the true path the dataset authors drew (dark) — 456 m apart |
| `green_mask.png` | that drawn path isolated, used to fit the correct transform |
| `munfrl_oriented.png` | after fixing query attitude, still using the wrong ground-truth frame |
| `munfrl_fixed.png`, `munfrl_final.png` | intermediate states as the transform was corrected |
| `munfrl_clean.png` | georeferencing correct, but the painted flight path inpainted out, leaving a smear across 23% of each tile |
| `esri_vs_munfrl.png` | why we switched to fetched satellite imagery instead |

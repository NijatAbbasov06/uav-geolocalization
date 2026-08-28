# Bounding VIO Drift with Visual Geo-Localization

Real-time, CPU-only absolute positioning for long-range GNSS-denied UAV flight.

Visual-inertial odometry drifts without bound: it integrates motion, so its error grows
with distance travelled and nothing ever pulls it back. Over a multi-kilometre flight with
no GNSS, that is fatal. This project pairs a VIO front-end with a self-supervised aerial
image-retrieval system that matches the downward-facing camera against pre-loaded satellite
imagery, producing **absolute** position fixes that bound the drift.

Everything is constrained to run in real time on **CPU only** — no GPU on the airframe.

---

## System

| Component | Role | Repository |
|---|---|---|
| **CAEVL** | Self-supervised geo-localization. Matches a UAV frame against a satellite tile database and returns an absolute position. | [`caevl`](https://github.com/NijatAbbasov06/caevl) |
| **OpenVINS** | MSCKF visual-inertial odometry. Provides smooth, high-rate relative motion. | [`open_vins`](https://github.com/NijatAbbasov06/open_vins) (branch `alto_integration`) |
| **Fusion** | Consensus outlier rejection + causal similarity-transform estimator. | [`tools/fuse.py`](tools/fuse.py) |

The two are complementary: VIO is locally accurate but drifts; geo-localization is globally
correct but noisy and intermittent. Fusion keeps the strengths of each.

---

## How CAEVL works

The hard part of matching a UAV photo to a satellite tile is the **domain gap** — different
sensors, seasons, times of day, sun angles. CAEVL sidesteps it by throwing away appearance
entirely and keeping only structure:

1. **Canny edge maps.** Both UAV and satellite images are reduced to edges. Roads, field
   boundaries and building outlines survive; colour, illumination and season do not.
2. **Stage 1 — autoencoder.** An encoder compresses an edge map to a 1024-d descriptor and a
   decoder reconstructs it. Optionally supervised by a DINOv2 perceptual loss.
3. **Stage 2 — VICReg fine-tuning.** The encoder is refined on (clean tile, degraded copy)
   pairs using the invariance/variance/covariance objective, which needs no negative pairs.

Crucially, **training uses reference imagery only** — no UAV photographs and no ground-truth
correspondences. That is what makes it deployable over new terrain: you need satellite
tiles for the area, nothing more.

At inference the tile descriptors are precomputed once; a query is one forward pass plus a
nearest-neighbour search.

---

## Results

### Reproduction — ViLD flight 10, held-out test split

| Configuration | R@1 @100 m | R@20 @100 m |
|---|---|---|
| Paper (reported) | 49.68% | — |
| **Ours, reproduction** | **48.14%** | 90.64% |
| **Ours + DINOv2 perceptual loss** | **52.88%** | 93.78% |
| Trained on both flights | 42.45% | 90.20% |

The baseline reproduces the paper within ~1.5 points.

### Perceptual-loss ablation

Adding the DINOv2 perceptual loss to stage 1 is worth **+4.7 points** R@1 (48.14 → 52.88).
This confirms the direction of the paper's Table 3, which reports a larger +8.2.

Training on both ViLD flights *hurt* (−5.7 points versus flight 10 alone) — independent
evidence that the model is sensitive to ground-sample distance, since the two flights differ
in altitude.

### Cross-dataset zero-shot — ALTO

The ViLD-trained encoder was applied to ALTO with no fine-tuning:

| | R@1 @50 m | R@1 @100 m | R@20 @100 m |
|---|---|---|---|
| Zero-shot | 55.46% | 62.05% | 75.89% |
| Zero-shot (perceptual) | 55.34% | 65.80% | 77.55% |

More than half of queries localize within 50 m on unseen terrain, from a model that never
saw a single ALTO image.

But **R@20 plateaus around 78%** — roughly a fifth of queries are never retrieved at any
depth. This is a **scale gap**, not a capacity limit: ViLD tiles cover a 1438 m footprint
against ALTO's 304 m, a factor of 4.7. Training a stronger model moved the ceiling by only
1.5 points, which is what pointed at scale rather than model quality.

### Runtime

**13.3 Hz** end-to-end on CPU, verified with `CUDA_VISIBLE_DEVICES=""`. Comfortably above
the rate the fusion filter needs, since fixes are only consumed intermittently.

### Confidence signals do not predict failure

Neither cosine similarity nor the top-1/top-2 margin separates good matches from bad — a
retrieval that was **23.7 km** wrong scored 0.958. Outlier rejection therefore cannot come
from the retrieval side; it has to come from the VIO prior via chi-square gating. This is
the single most important design constraint on the fusion layer.

---

## End-to-end result — MUN-FRL `bell412_dataset4`

ALTO ships no IMU, so the full system could not run on it. The pipeline was moved to
[MUN-FRL](https://mun-frl-vil-dataset.readthedocs.io/), the only public aerial dataset found
that provides a downward camera, a hardware-synchronised IMU **and** RTK ground truth.
`bell412_dataset4` is 3.6 km over 319 s, flown from a Bell 412 over Ottawa International
Airport.

| | OpenVINS | Fused | |
|---|---|---|---|
| **Causal (real-time)** RMSE | 617 m | **392 m** | −36% |
| Offline, whole flight | 452 m | 314 m | −31% |
| Offline, cruise only | 166 m | **48 m** | −71% |

Both trajectories anchored at takeoff. The causal estimator uses only information available
at each instant; the offline figures are an upper bound, reported separately rather than
conflated.

### Consensus outlier rejection

The prediction stream is 82% wrong, and — as the ALTO work already showed — the retrieval
score cannot identify which. Measured over two flights it is *slightly negatively*
correlated with correctness (0.9429 when right, 0.9493 when wrong).

Incorrect predictions scatter roughly uniformly over the tile database, so two of them
agreeing — after accounting for the VIO displacement between their timestamps — is very
unlikely. Screening a sliding window with RANSAC on that basis:

| | before | after |
|---|---|---|
| predictions | 299 | 101 |
| precision | 17.7% | **99.0%** |
| correct predictions retained | 107 | 100 (93%) |

This never consults the retrieval score or the filter's own belief, so it avoids the
divergence mode in which a drifted filter rejects exactly the measurements that would fix it.

### Search restriction

Restricting the tile database to a corridor around positions where queries exist — in
deployment, the uncertainty region defined by the motion prior — cuts it from 5340 to 1612
tiles and doubles recall. 83% of erroneous predictions were matching tiles that could never
have been correct.

| configuration | tiles | R@100 m | median err | ms/query |
|---|---|---|---|---|
| 100 m tile, 400 m corridor | 5340 | 17.7% | 406 m | 229 |
| 100 m tile, 200 m corridor | 1612 | **35.8%** | **230 m** | 233 |
| 60 m tile, 200 m corridor | 3270 | 13.6% | 476 m | 235 |

Shrinking tiles to 60 m to admit lower-altitude frames **fails**: 2.21% of tile pairs then
exceed 0.99 cosine similarity against 0.31% at 100 m, so distinct places become
indistinguishable. Runtime is flat across a 3.3× database range — the cost is encoding the
query, not searching the map.

### Terrain dependence

Same aircraft, same airport, same weights: `dataset3` scores 8.9% R@100 m against
`dataset4`'s 17.7%. `dataset3` loops over continuous forest; `dataset4` crosses the terminal
apron and taxiways. Over uniform canopy every Canny edge map looks alike.

### A bug in the dataset

MUN-FRL's shipped map metadata places the flight **456 m** from where it actually occurred.
A bounds check cannot catch it — the track still lands inside the image — but every extracted
tile then depicts different ground than it claims. `tools/munfrl_map_tiles.py` detects and
corrects this automatically by fitting the ground-truth track onto the flight path the
dataset authors drew on each map (residual 0.35 px).

### Limitations

- **Altitude coverage is the binding constraint.** A 100 m tile needs ~66 m of altitude to be
  fully imaged; only 34% of this flight is above that. The profile is takeoff, ~70 s of
  cruise at 105 m, descent, then 80 s at ~11 m where the camera sees 17 m of ground. No
  map-matching method can localise from that.
- **Two quantities come from ground truth**: query altitude, and the VIO-to-map yaw offset.
  In deployment these come from an altimeter and a magnetometer or first fix; MUN-FRL records
  neither. The VIO's own altitude is not a substitute (median error 29.8 m).
- Fusion evaluated on one flight, retrieval on two.
- The VIO's ~40% scale deficit is unexplained; IMU restamping did not remove it.

---

## VIO baseline

**EuRoC MH_01_easy** — validated end to end, evaluated with `ov_eval` after `posyaw`
alignment (translation + yaw are unobservable gauge freedoms in VIO; removing them is
standard, and removing *more* than them would hide real error).

**ALTO** ships no IMU, so OpenVINS cannot run on it directly. The trajectory is instead
driven through OpenVINS's simulator, which B-splines the ground-truth path and synthesizes
IMU and feature measurements from it.

The ALTO flights are near-perfect straight lines — 4593 m of path over a 4590 m straight-line
distance, with 7° of heading swept in total. Characterising VIO behaviour under that motion
profile is ongoing work.

---

## Report and slides

| | |
|---|---|
| [`report/report.pdf`](report/report.pdf) | 9 pages, IEEE format, 15 figures |
| [`report/slides.pdf`](report/slides.pdf) | 10 slides |
| [`report/slides_turkish.pdf`](report/slides_turkish.pdf) | the same deck in Turkish |

Sources sit alongside them; rebuild with `pdflatex` + `biber`.

## Plots

[`diagnostics/`](diagnostics/) holds every figure generated during the work — the
15 used in the report plus the intermediate ones that record how the 456 m
georeferencing bug was found. [`diagnostics/README.md`](diagnostics/README.md)
indexes them.

---

## Repository layout

This repo holds the report, the fusion and data-preparation tools, and the figures. The
two upstream systems live in separate forks so that the diff against each stays visible and
attribution is preserved.

- **[`report/`](report/)** — IEEE-format write-up (`report.tex`, `references.bib`) and figures.
- **[`tools/`](tools/)** — data preparation and fusion:
  `restamp_imu.py` (IMU hardware-clock correction, jitter ↓37×),
  `ppk_to_ovgt.py` (RTKLIB → ASL ground truth),
  `fetch_basemap.py` (Esri imagery download + ENU→pixel solve),
  `munfrl_map_tiles.py` (tile database, self-calibrating georeference),
  `make_queries_oriented.py` (attitude-corrected ground-plane resampling),
  `fuse.py` (consensus rejection + fusion; `--mode causal | align | ekf`).
- **[`config/`](config/)** — OpenVINS configuration for MUN-FRL.

- **[caevl](https://github.com/NijatAbbasov06/caevl)** — training configs for every
  ablation, retrieval evaluation, interactive result explorers, `localizer.py` (CPU runtime
  inference), tile preparation tools.
- **[open_vins](https://github.com/NijatAbbasov06/open_vins)** — branch `alto_integration`:
  ALTO simulator configs, `ov_eval/scripts/record_trajectory.py` (a ROS2 replacement for the
  ROS1-only `pose_to_file` recorder), launch fixes for ground-truth plotting.

---

## Credits

- **CAEVL** — [Tristan-Amadei/caevl](https://github.com/Tristan-Amadei/caevl)
- **OpenVINS** — [rpng/open_vins](https://github.com/rpng/open_vins), GPLv3
- **ALTO** — Cartographer/ALTO aerial localization dataset, CMU
- **ViLD** — Visual Localization Dataset
- **EuRoC MAV** — ETH Zurich

Each fork retains its upstream license.

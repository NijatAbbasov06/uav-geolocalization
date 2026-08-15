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
| **Fusion** | Loosely-coupled filter consuming VIO deltas + gated CAEVL fixes. | *in progress* |

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

## Repository layout

This repo is the front door. The code lives in two forks, kept separate so that the diff
against each upstream project stays visible and attribution is preserved:

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

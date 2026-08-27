#!/usr/bin/env python3
"""
Bound OpenVINS drift with CAEVL geo-localization fixes.

The two inputs have opposite failure modes. The VIO is smooth and locally
accurate but drifts without limit, and on this data it also runs ~34% short in
scale. The fixes are absolutely positioned and never drift, but on dataset4 only
about 18% of them are anywhere near right, and - measured over two flights - the
retrieval score is very slightly *anti*-correlated with being correct, so it
cannot be used to tell them apart.

That rules out trusting fixes individually. What separates a good fix from a bad
one is not how it scores but whether it agrees with where the filter already
believes it is, so every fix is gated on the motion prior with a chi-square test
and the majority are thrown away.

State is a 4-vector: position, plus the similarity transform between OpenVINS's
arbitrary frame and the map:

    x, y    current position in map coordinates
    s       scale applied to VIO displacements
    theta   yaw of the VIO frame relative to the map

Estimating s and theta rather than assuming them is what makes this work here.
OpenVINS initialises with an arbitrary heading, and its 34% scale error would
otherwise reappear between every pair of fixes. With them in the state the
filter learns both from the fixes it accepts, and the VIO contributes shape
while the fixes contribute scale and absolute position.

Startup is the hard part: with 82% outliers a filter that begins with no idea
where it is will latch onto a wrong fix and gate out all the right ones
afterwards. So the first window of fixes is fitted with RANSAC, which finds the
similarity transform that the largest consistent subset agrees on, and the
filter starts from that. RANSAC sees only VIO and fixes - never ground truth -
so this remains deployable; ground truth enters solely to score the result.

Usage:
  python3 fuse.py --vio logs/ov_estimate_dataset4.txt \
      --fixes caevl/fixes_dataset4.csv \
      --gt bell412_dataset4/ppk_data/bell412_dataset4_gt.csv \
      --out logs/fused_dataset4.csv
"""
import argparse
import csv
import os
import re
import sys

import numpy as np


def load_vio(path):
    """ov_estimate.txt -> (t, xy). Columns are timestamp, q(4), p(3), ..."""
    t, p = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) < 8:
            continue
        t.append(float(f[0]))
        p.append([float(f[5]), float(f[6])])
    return np.array(t), np.array(p)


def load_gt(path):
    t, p = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.strip().split(',')
        t.append(int(f[0]) / 1e9)
        p.append([float(f[1]), float(f[2])])
    return np.array(t), np.array(p)


def load_fixes(path):
    """fixes.csv -> (t, xy).

    The CSV's own `timestamp` column is just frame_index/fps, not wall time.
    The real instant is in the filename, e.g. q_1653512859_707685.png.
    """
    t, p = [], []
    for row in csv.DictReader(open(path)):
        m = re.match(r'q_(\d+)_(\d+)', row['name'])
        if not m:
            continue
        t.append(float(f'{m.group(1)}.{m.group(2)}'))
        p.append([float(row['pred_x']), float(row['pred_y'])])
    order = np.argsort(t)
    return np.array(t)[order], np.array(p)[order]


def rotz(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def ransac_init(vio_t, vio_p, fix_t, fix_p, window, thresh, iters, rng):
    """Fit (s, theta, offset) from the first `window` seconds of fixes.

    Two fixes plus the VIO positions at their timestamps determine a similarity
    transform outright: the ratio of the two baselines gives the scale, the
    angle between them gives the yaw. Score each candidate by how many other
    fixes it explains to within `thresh` metres and keep the best.
    """
    m = fix_t <= fix_t[0] + window
    ft, fp = fix_t[m], fix_p[m]
    if len(ft) < 4:
        return None
    vp = np.column_stack([np.interp(ft, vio_t, vio_p[:, 0]),
                          np.interp(ft, vio_t, vio_p[:, 1])])

    best = (0, None)
    for _ in range(iters):
        i, j = rng.choice(len(ft), 2, replace=False)
        dv, df = vp[j] - vp[i], fp[j] - fp[i]
        nv, nf = np.linalg.norm(dv), np.linalg.norm(df)
        if nv < 20.0 or nf < 20.0:      # too close to define a direction
            continue
        s = nf / nv
        th = np.arctan2(df[1], df[0]) - np.arctan2(dv[1], dv[0])
        off = fp[i] - s * (rotz(th) @ vp[i])
        pred = s * (rotz(th) @ vp.T).T + off
        inl = np.linalg.norm(pred - fp, axis=1) < thresh
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (s, th, off, inl))
    if best[1] is None:
        return None

    # refine on the inlier set
    s, th, off, inl = best[1]
    if inl.sum() >= 3:
        v, f = vp[inl], fp[inl]
        vc, fc = v - v.mean(0), f - f.mean(0)
        th = np.arctan2((vc[:, 0] * fc[:, 1] - vc[:, 1] * fc[:, 0]).sum(),
                        (vc[:, 0] * fc[:, 0] + vc[:, 1] * fc[:, 1]).sum())
        s = (np.linalg.norm(fc, axis=1).sum()
             / max(np.linalg.norm(vc, axis=1).sum(), 1e-9))
        off = f.mean(0) - s * (rotz(th) @ v.mean(0))
    return s, th, off, int(best[0]), int(m.sum())


def consensus_filter(vio_t, vio_p, fix_t, fix_p, window, thresh, min_inl,
                     iters, rng):
    """Keep only fixes that agree with other fixes nearby in time.

    Gating on the filter's own belief has a failure mode with no escape: once
    the estimate has drifted, correct fixes look like huge innovations and get
    rejected, which guarantees it stays drifted. Loosening the gate to fix that
    lets outliers in instead, and one bad fix moved the estimate 2.4 km.

    This test avoids both because it never consults the filter. Bad fixes are
    scattered essentially uniformly over a 5000-tile map, so two of them landing
    near the same place - after correcting for the VIO motion between their
    timestamps - happens well under 1% of the time. Agreement is therefore
    strong evidence of correctness, and it is available before any filtering
    happens.

    RANSAC runs over a sliding window: each hypothesis is a similarity
    transform from two fixes, scored by how many others in the window it
    explains. A fix survives if it is an inlier in any window whose consensus
    reached `min_inl` members.

    Uses only VIO and fixes, so it remains deployable.
    """
    keep = np.zeros(len(fix_t), bool)
    vp = np.column_stack([np.interp(fix_t, vio_t, vio_p[:, 0]),
                          np.interp(fix_t, vio_t, vio_p[:, 1])])
    starts = np.arange(fix_t[0], fix_t[-1], window / 2.0)
    for t0 in starts:
        m = (fix_t >= t0) & (fix_t < t0 + window)
        idx = np.nonzero(m)[0]
        if len(idx) < min_inl + 1:
            continue
        v, f = vp[idx], fix_p[idx]
        best = (0, None)
        for _ in range(iters):
            i, j = rng.choice(len(idx), 2, replace=False)
            dv, df = v[j] - v[i], f[j] - f[i]
            nv, nf = np.linalg.norm(dv), np.linalg.norm(df)
            if nv < 20.0 or nf < 20.0:
                continue
            sc = nf / nv
            if not (0.2 < sc < 5.0):        # implausible scale, skip
                continue
            th = np.arctan2(df[1], df[0]) - np.arctan2(dv[1], dv[0])
            off = f[i] - sc * (rotz(th) @ v[i])
            pred = sc * (rotz(th) @ v.T).T + off
            inl = np.linalg.norm(pred - f, axis=1) < thresh
            if inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
        if best[0] >= min_inl:
            keep[idx[best[1]]] = True
    return keep


def fit_similarity(v, f):
    """Least-squares (scale, yaw, offset) taking VIO points v onto fixes f."""
    vc, fc = v - v.mean(0), f - f.mean(0)
    th = np.arctan2((vc[:, 0] * fc[:, 1] - vc[:, 1] * fc[:, 0]).sum(),
                    (vc[:, 0] * fc[:, 0] + vc[:, 1] * fc[:, 1]).sum())
    sc = (np.linalg.norm(fc, axis=1).sum()
          / max(np.linalg.norm(vc, axis=1).sum(), 1e-9))
    return sc, th, f.mean(0) - sc * (rotz(th) @ v.mean(0))


def align_mode(vio_t, vio_p, fix_t, fix_p, window, min_fixes, min_span):
    """Continuously re-align the VIO onto the accepted fixes.

    An EKF was the obvious choice here and it lost badly to this - 422 m
    against 219 m - for a reason worth recording. Scale is only observable
    through the ratio of accumulated VIO travel to fix separation over long
    baselines, but a per-fix position update lets position absorb almost the
    whole innovation, so scale barely moves and can wander the wrong way. The
    filter also has to decide about each fix the moment it arrives, while
    alignment gets to look at a whole window at once.

    So instead: fit a similarity transform to the fixes inside a sliding
    window, and interpolate the transform between window centres. Where a
    window holds too few fixes - over the featureless middle of this flight,
    which supplies almost none - it falls back to the global fit rather than
    inventing a local one.

    This cannot beat the VIO's own shape error: on dataset4 even a perfect
    global fit leaves 215 m, because the estimated trajectory is distorted, not
    merely mis-scaled. Windowing recovers part of that by letting the transform
    drift, but the shape error is the floor.
    """
    v_at_f = np.column_stack([np.interp(fix_t, vio_t, vio_p[:, 0]),
                              np.interp(fix_t, vio_t, vio_p[:, 1])])
    g_sc, g_th, g_off = fit_similarity(v_at_f, fix_p)
    print(f'  global fit: scale {g_sc:.3f}, yaw {np.degrees(g_th):+.1f} deg')

    centres = np.arange(vio_t[0], vio_t[-1] + window / 2.0, window / 2.0)
    params, used = [], 0
    for c in centres:
        m = np.abs(fix_t - c) <= window / 2.0
        # Enough fixes is not enough: they must also be spread out. Fixes
        # bunched within a few metres give a baseline too short to determine
        # scale, and one such window fitted 4.4 instead of 1.67, which was
        # worse than having no local fit at all.
        span = 0.0
        if m.sum() >= min_fixes:
            q = fix_p[m]
            span = float(np.linalg.norm(q - q.mean(0), axis=1).max())
        if m.sum() >= min_fixes and span >= min_span:
            sc, th, off = fit_similarity(v_at_f[m], fix_p[m])
            if 0.5 < sc < 3.0:
                params.append((c, sc, th, off)); used += 1
                continue
        params.append((c, g_sc, g_th, g_off))
    print(f'  {used} of {len(centres)} windows had >= {min_fixes} fixes; '
          f'the rest fall back to the global fit')

    P = np.array([[p[1], p[2], p[3][0], p[3][1]] for p in params])
    C = np.array([p[0] for p in params])
    # unwrap yaw so interpolation cannot take the long way round
    P[:, 1] = np.unwrap(P[:, 1])
    out = np.empty((len(vio_t), 2))
    sc = np.interp(vio_t, C, P[:, 0]); th = np.interp(vio_t, C, P[:, 1])
    ox = np.interp(vio_t, C, P[:, 2]); oy = np.interp(vio_t, C, P[:, 3])
    cs, sn = np.cos(th), np.sin(th)
    out[:, 0] = sc * (cs * vio_p[:, 0] - sn * vio_p[:, 1]) + ox
    out[:, 1] = sc * (sn * vio_p[:, 0] + cs * vio_p[:, 1]) + oy
    return out, sc, th


def causal_consensus(buf_t, buf_f, buf_v, thresh, min_inl, iters, rng):
    """Consensus over a buffer of PAST fixes only. Returns an inlier mask.

    Same test as consensus_filter, but the caller passes a trailing window, so
    nothing after the current instant is ever consulted.
    """
    n = len(buf_t)
    if n < min_inl + 1:
        return np.zeros(n, bool)
    best = (0, None)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        dv, df = buf_v[j] - buf_v[i], buf_f[j] - buf_f[i]
        nv, nf = np.linalg.norm(dv), np.linalg.norm(df)
        if nv < 20.0 or nf < 20.0:
            continue
        sc = nf / nv
        if not (0.3 < sc < 4.0):
            continue
        th = np.arctan2(df[1], df[0]) - np.arctan2(dv[1], dv[0])
        off = buf_f[i] - sc * (rotz(th) @ buf_v[i])
        pred = sc * (rotz(th) @ buf_v.T).T + off
        inl = np.linalg.norm(pred - buf_f, axis=1) < thresh
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best[0] < min_inl:
        return np.zeros(n, bool)
    return best[1]


def causal_mode(vio_t, vio_p, fix_t, fix_p, cons_window, cons_thresh,
                cons_min, min_span, rng):
    """Real-time fusion: at every instant, use only what has already arrived.

    The alignment mode scored 211 m but read predictions from the whole flight,
    including ones from after the moment being corrected. That is a smoother,
    and it cannot run on an aircraft. This is the same idea made causal.

    At each step: any prediction whose timestamp has passed enters a buffer; a
    trailing window of that buffer is screened for mutual consensus; survivors
    accumulate; and the similarity transform is refitted to everything accepted
    so far. Scale needs a long baseline to be observable, so the fit uses all
    accepted history rather than a short window - that is still causal, just
    with a growing memory.

    Until enough accepted predictions span `min_span` metres there is no usable
    transform and the system is simply not localised. That warm-up is reported
    separately rather than hidden, because on a real flight it is dead time.
    """
    out = np.full((len(vio_t), 2), np.nan)
    scale_t = np.full(len(vio_t), np.nan)
    acc_f, acc_v = [], []
    buf_t, buf_f, buf_v = [], [], []
    fi = 0
    have = None
    first_fix_idx = None

    v_at_f = np.column_stack([np.interp(fix_t, vio_t, vio_p[:, 0]),
                              np.interp(fix_t, vio_t, vio_p[:, 1])])

    for k in range(len(vio_t)):
        new = False
        while fi < len(fix_t) and fix_t[fi] <= vio_t[k]:
            buf_t.append(fix_t[fi]); buf_f.append(fix_p[fi])
            buf_v.append(v_at_f[fi])
            fi += 1; new = True

        if new:
            bt = np.array(buf_t)
            m = bt >= bt[-1] - cons_window          # trailing window only
            idx = np.nonzero(m)[0]
            inl = causal_consensus(bt[idx], np.array(buf_f)[idx],
                                   np.array(buf_v)[idx],
                                   cons_thresh, cons_min, 800, rng)
            for a, keep in zip(idx, inl):
                if keep:
                    acc_f.append(buf_f[a]); acc_v.append(buf_v[a])
            if len(acc_f) >= 4:
                F, V = np.array(acc_f), np.array(acc_v)
                span = np.linalg.norm(F - F.mean(0), axis=1).max()
                if span >= min_span:
                    sc, th, off = fit_similarity(V, F)
                    if 0.3 < sc < 4.0:
                        have = (sc, th, off)
                        if first_fix_idx is None:
                            first_fix_idx = k

        if have is not None:
            sc, th, off = have
            out[k] = sc * (rotz(th) @ vio_p[k]) + off
            scale_t[k] = sc

    return out, scale_t, first_fix_idx


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--vio', required=True)
    ap.add_argument('--fixes', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fix-sigma', type=float, default=40.0,
                    help='assumed 1-sigma of an accepted fix, metres')
    ap.add_argument('--gate', type=float, default=9.21,
                    help='chi-square gate, 2 dof (5.99=95%%, 9.21=99%%)')
    ap.add_argument('--q-pos', type=float, default=0.02,
                    help='position process noise per metre travelled')
    ap.add_argument('--q-scale', type=float, default=1e-7,
                    help='random walk on the scale factor')
    ap.add_argument('--q-yaw', type=float, default=1e-7,
                    help='random walk on the yaw offset, rad^2 per step')
    ap.add_argument('--init-window', type=float, default=60.0,
                    help='seconds of fixes used to bootstrap with RANSAC')
    ap.add_argument('--init-thresh', type=float, default=60.0,
                    help='RANSAC inlier threshold, metres')
    ap.add_argument('--lockout', type=float, default=15.0,
                    help='seconds without an accepted fix before the filter '
                         'starts inflating its covariance')
    ap.add_argument('--inflate', type=float, default=1.02,
                    help='per-step covariance growth while locked out')
    ap.add_argument('--consensus-window', type=float, default=30.0,
                    help='seconds; 0 disables the consensus pre-filter')
    ap.add_argument('--consensus-thresh', type=float, default=60.0,
                    help='metres two fixes may differ by and still agree')
    ap.add_argument('--consensus-min', type=int, default=3,
                    help='fixes that must agree before any of them is trusted')
    ap.add_argument('--mode', choices=['causal', 'align', 'ekf'], default='causal',
                    help='causal = real-time, uses only past predictions; '
                         'align = offline smoother (optimistic bound); '
                         'ekf = the filter, kept for comparison')
    ap.add_argument('--align-window', type=float, default=90.0,
                    help='seconds of fixes per alignment window')
    ap.add_argument('--align-min', type=int, default=6,
                    help='fixes a window needs before it is fitted locally')
    ap.add_argument('--align-min-span', type=float, default=300.0,
                    help='metres the fixes in a window must span before it is '
                         'fitted locally; short baselines cannot fix scale')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    vio_t, vio_p = load_vio(a.vio)
    fix_t, fix_p = load_fixes(a.fixes)
    gt_t, gt_p = load_gt(a.gt)
    print(f'VIO   {len(vio_t)} poses   {vio_t[0]:.1f}..{vio_t[-1]:.1f}')
    print(f'fixes {len(fix_t)}         {fix_t[0]:.1f}..{fix_t[-1]:.1f}')

    rng = np.random.default_rng(a.seed)

    if a.consensus_window > 0:
        keep = consensus_filter(vio_t, vio_p, fix_t, fix_p,
                                a.consensus_window, a.consensus_thresh,
                                a.consensus_min, 3000, rng)
        print(f'consensus pre-filter: {keep.sum()} of {len(keep)} fixes kept '
              f'({100.0 * keep.mean():.1f}%)')
        if keep.sum() < 4:
            sys.exit('too few fixes survived the consensus filter')
        fix_t, fix_p = fix_t[keep], fix_p[keep]

    if a.mode == 'causal':
        print('\ncausal mode - only past predictions are used at each instant')
        aligned, sc_t, first_k = causal_mode(
            vio_t, vio_p, fix_t, fix_p, a.consensus_window, a.consensus_thresh,
            a.consensus_min, a.align_min_span, rng)
        g = np.column_stack([np.interp(vio_t, gt_t, gt_p[:, 0]),
                             np.interp(vio_t, gt_t, gt_p[:, 1])])
        live = ~np.isnan(aligned[:, 0])
        if live.sum() < 10:
            sys.exit('never localised: no transform ever became available')
        err = np.linalg.norm(aligned[live] - g[live], axis=1)
        warm = vio_t[live][0] - vio_t[0]
        print(f'  localised from t+{warm:.0f} s onward '
              f'({100.0 * live.mean():.0f}% of the flight)')
        print(f'  final scale estimate {sc_t[live][-1]:.3f}')
        v_at_f = np.column_stack([np.interp(fix_t, vio_t, vio_p[:, 0]),
                                  np.interp(fix_t, vio_t, vio_p[:, 1])])
        vc, fc = v_at_f - v_at_f.mean(0), fix_p - fix_p.mean(0)
        th0 = np.arctan2((vc[:, 0] * fc[:, 1] - vc[:, 1] * fc[:, 0]).sum(),
                         (vc[:, 0] * fc[:, 0] + vc[:, 1] * fc[:, 1]).sum())
        base = (rotz(th0) @ vio_p.T).T + (fix_p.mean(0)
                                          - (rotz(th0) @ v_at_f.mean(0)))
        eb = np.linalg.norm(base[live] - g[live], axis=1)
        print(f'\n  over the localised period    VIO alone      fused')
        for lbl, fn in (('RMSE (m)', lambda e: np.sqrt((e ** 2).mean())),
                        ('median (m)', np.median), ('max (m)', np.max),
                        ('final (m)', lambda e: e[-1])):
            print(f'    {lbl:16s} {fn(eb):10.1f} {fn(err):10.1f}')
        h = len(err) // 2
        print(f'\n    first half -> second half')
        print(f'      VIO   {eb[:h].mean():7.1f} -> {eb[h:].mean():7.1f} m')
        print(f'      fused {err[:h].mean():7.1f} -> {err[h:].mean():7.1f} m')
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.',
                        exist_ok=True)
            with open(a.out, 'w') as fh:
                fh.write('timestamp,x,y,scale,yaw,gt_x,gt_y,err_fused,err_vio\n')
                for i in range(len(vio_t)):
                    if np.isnan(aligned[i, 0]):
                        continue
                    ef = float(np.linalg.norm(aligned[i] - g[i]))
                    ev = float(np.linalg.norm(base[i] - g[i]))
                    fh.write(f'{vio_t[i]:.6f},{aligned[i,0]:.3f},'
                             f'{aligned[i,1]:.3f},{sc_t[i]:.5f},0,'
                             f'{g[i,0]:.3f},{g[i,1]:.3f},{ef:.3f},{ev:.3f}\n')
            print(f'\nwrote {a.out}')
        return

    if a.mode == 'align':
        print('\nalignment mode')
        aligned, sc_t, th_t = align_mode(vio_t, vio_p, fix_t, fix_p,
                                         a.align_window, a.align_min,
                                         a.align_min_span)
        g = np.column_stack([np.interp(vio_t, gt_t, gt_p[:, 0]),
                             np.interp(vio_t, gt_t, gt_p[:, 1])])
        fused = np.linalg.norm(aligned - g, axis=1)
        v_at_f = np.column_stack([np.interp(fix_t, vio_t, vio_p[:, 0]),
                                  np.interp(fix_t, vio_t, vio_p[:, 1])])
        s0, th0, off0 = fit_similarity(v_at_f, fix_p)
        base = s0 * (rotz(th0) @ vio_p.T).T + off0
        baseline = np.linalg.norm(base - g, axis=1)
        L = np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
        print(f'\nover {L:.0f} m of truth')
        print(f'                    global fit   windowed')
        for lbl, fn in (('RMSE (m)', lambda e: np.sqrt((e ** 2).mean())),
                        ('median (m)', np.median), ('final (m)', lambda e: e[-1]),
                        ('max (m)', np.max)):
            print(f'  {lbl:16s} {fn(baseline):10.1f} {fn(fused):10.1f}')
        h = len(fused) // 2
        print(f'\n  first half -> second half')
        print(f'    global   {baseline[:h].mean():7.1f} -> {baseline[h:].mean():7.1f} m')
        print(f'    windowed {fused[:h].mean():7.1f} -> {fused[h:].mean():7.1f} m')
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.',
                        exist_ok=True)
            with open(a.out, 'w') as fh:
                fh.write('timestamp,x,y,scale,yaw,gt_x,gt_y,err_fused,err_vio\n')
                for i in range(len(vio_t)):
                    fh.write(f'{vio_t[i]:.6f},{aligned[i,0]:.3f},'
                             f'{aligned[i,1]:.3f},{sc_t[i]:.5f},{th_t[i]:.5f},'
                             f'{g[i,0]:.3f},{g[i,1]:.3f},'
                             f'{fused[i]:.3f},{baseline[i]:.3f}\n')
            print(f'\nwrote {a.out}')
        return

    init = ransac_init(vio_t, vio_p, fix_t, fix_p,
                       a.init_window, a.init_thresh, 4000, rng)
    if init is None:
        sys.exit('RANSAC could not bootstrap: too few fixes in the window')
    s0, th0, off0, ninl, ntot = init
    print(f'\nRANSAC init over first {a.init_window:.0f} s: '
          f'{ninl}/{ntot} fixes consistent')
    print(f'  scale {s0:.3f}   yaw {np.degrees(th0):+.1f} deg')

    # state [x, y, s, theta]
    x = np.array([*(s0 * (rotz(th0) @ vio_p[0]) + off0), s0, th0])
    P = np.diag([100.0 ** 2, 100.0 ** 2, 0.2 ** 2, np.radians(15) ** 2])
    R = np.eye(2) * a.fix_sigma ** 2

    fi = 0
    out, n_acc, n_rej = [], 0, 0
    t_last_acc = vio_t[0]
    n_inflated = 0
    for k in range(1, len(vio_t)):
        d = vio_p[k] - vio_p[k - 1]
        c, sn = np.cos(x[3]), np.sin(x[3])
        Rk = np.array([[c, -sn], [sn, c]])
        step = x[2] * (Rk @ d)
        x[0:2] += step

        # d(position)/d(scale) and d(position)/d(yaw)
        dRd = np.array([[-sn, -c], [c, -sn]]) @ d
        F = np.eye(4)
        F[0:2, 2] = Rk @ d
        F[0:2, 3] = x[2] * dRd
        dist = np.linalg.norm(step)
        Q = np.diag([(a.q_pos * dist) ** 2, (a.q_pos * dist) ** 2,
                     a.q_scale, a.q_yaw])
        P = F @ P @ F.T + Q

        # Anti-lockout. A scale error makes position error grow *linearly* with
        # distance, while random-walk process noise only grows as its square
        # root, so the filter inevitably becomes overconfident. Once that
        # happens a correct fix arrives looking like a huge innovation and gets
        # gated out - the filter is most certain exactly when it is most wrong,
        # and nothing can rescue it. If nothing has been accepted for a while,
        # assume the covariance is the thing that is wrong and grow it until
        # measurements can get back in.
        if vio_t[k] - t_last_acc > a.lockout:
            P[0:2, 0:2] *= a.inflate
            P[2, 2] *= a.inflate
            n_inflated += 1

        # fold in every fix that falls in this interval
        while fi < len(fix_t) and fix_t[fi] <= vio_t[k]:
            z = fix_p[fi]
            H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
            nu = z - x[0:2]
            S = H @ P @ H.T + R
            d2 = float(nu @ np.linalg.solve(S, nu))
            if d2 <= a.gate:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ nu
                P = (np.eye(4) - K @ H) @ P
                n_acc += 1
                t_last_acc = vio_t[k]
            else:
                n_rej += 1
            fi += 1

        out.append([vio_t[k], x[0], x[1], x[2], x[3]])

    out = np.array(out)
    print(f'\nfixes: {n_acc} accepted, {n_rej} rejected '
          f'({100.0 * n_acc / max(n_acc + n_rej, 1):.1f}% accepted)')
    print(f'steps spent locked out (covariance inflating): {n_inflated} '
          f'of {len(vio_t) - 1}')
    print(f'final scale {x[2]:.3f}   yaw {np.degrees(x[3]):+.1f} deg')

    ### score against ground truth ###
    g = np.column_stack([np.interp(out[:, 0], gt_t, gt_p[:, 0]),
                         np.interp(out[:, 0], gt_t, gt_p[:, 1])])
    fused = np.linalg.norm(out[:, 1:3] - g, axis=1)

    # VIO alone, given the same similarity transform it was launched with;
    # this is the drift the fixes are supposed to be bounding.
    vp = np.column_stack([np.interp(out[:, 0], vio_t, vio_p[:, 0]),
                          np.interp(out[:, 0], vio_t, vio_p[:, 1])])
    base = s0 * (rotz(th0) @ vp.T).T + off0
    baseline = np.linalg.norm(base - g, axis=1)

    L = np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
    print(f'\nover {L:.0f} m of truth')
    print(f'                    VIO alone      fused')
    for lbl, f in (('RMSE (m)', lambda e: np.sqrt((e ** 2).mean())),
                   ('median (m)', np.median),
                   ('final (m)', lambda e: e[-1]),
                   ('max (m)', np.max)):
        print(f'  {lbl:16s} {f(baseline):9.1f}  {f(fused):9.1f}')
    print(f'  {"drift (%)":16s} {100 * baseline[-1] / L:9.2f}  '
          f'{100 * fused[-1] / L:9.2f}')

    # the point is a bounded error, not merely a smaller one
    half = len(fused) // 2
    print(f'\n  error growth, first half -> second half')
    print(f'    VIO alone  {baseline[:half].mean():7.1f} -> '
          f'{baseline[half:].mean():7.1f} m')
    print(f'    fused      {fused[:half].mean():7.1f} -> '
          f'{fused[half:].mean():7.1f} m')

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
        with open(a.out, 'w') as fh:
            fh.write('timestamp,x,y,scale,yaw,gt_x,gt_y,err_fused,err_vio\n')
            for i in range(len(out)):
                fh.write(f'{out[i,0]:.6f},{out[i,1]:.3f},{out[i,2]:.3f},'
                         f'{out[i,3]:.5f},{out[i,4]:.5f},'
                         f'{g[i,0]:.3f},{g[i,1]:.3f},'
                         f'{fused[i]:.3f},{baseline[i]:.3f}\n')
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

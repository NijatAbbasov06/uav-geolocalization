#!/usr/bin/env python3
"""
Solve the position+yaw transform between the OpenVINS estimate and the ENU
ground truth, and rewrite the ground-truth CSV into the OpenVINS frame so the
two overlay in RViz.

VIO leaves global position and heading unobservable (roll/pitch are fixed by
gravity, scale is metric). So the only legitimate alignment is a translation
plus a rotation about z -- "posyaw". We deliberately do NOT solve for scale:
a Sim(3) fit would absorb scale error into the transform and make a run with
a wrong trajectory length look correct.

  estimate file : filepath_est from estimator_config.yaml
                  "# timestamp(s) q p v bg ba ..."  -> t=col0, p=cols 5,6,7
  gt file       : output of ppk_to_ovgt.py (ENU, origin at first PPK epoch)

By default the fit uses only the first --window seconds, so that a run which
diverges later does not drag the alignment with it.

Usage:
  python3 align_gt.py <ov_estimate.txt> <gt_enu.csv> <gt_aligned.csv> [--window 30]
"""
import argparse, math


def read_est(path):
    out = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) < 8:
            continue
        out.append((float(f[0]), float(f[5]), float(f[6]), float(f[7])))
    return out


def read_gt(path):
    out = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split(',')
        out.append((int(f[0]) / 1e9, float(f[1]), float(f[2]), float(f[3])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('est'); ap.add_argument('gt'); ap.add_argument('dst')
    ap.add_argument('--window', type=float, default=30.0,
                    help='seconds of the run to fit over (0 = all)')
    a = ap.parse_args()

    est, gt = read_est(a.est), read_gt(a.gt)
    if not est:
        raise SystemExit(f'no poses in {a.est} -- is save_total_state true?')
    print(f'estimate: {len(est)} poses, t {est[0][0]:.3f}..{est[-1][0]:.3f}')
    print(f'gt      : {len(gt)} poses, t {gt[0][0]:.3f}..{gt[-1][0]:.3f}')

    # pair each estimate pose with a linearly interpolated gt pose
    cut = est[0][0] + a.window if a.window > 0 else float('inf')
    pairs, j = [], 0
    for t, ex, ey, ez in est:
        if t > cut:
            break
        while j + 2 < len(gt) and gt[j + 1][0] < t:
            j += 1
        t0, t1 = gt[j][0], gt[j + 1][0]
        if not (t0 <= t <= t1):
            continue
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        g = tuple(gt[j][k] + w * (gt[j + 1][k] - gt[j][k]) for k in (1, 2, 3))
        pairs.append((g, (ex, ey, ez)))
    if len(pairs) < 10:
        raise SystemExit(f'only {len(pairs)} overlapping poses -- check the timebase')
    print(f'fitting on {len(pairs)} pose pairs over {a.window or "all"} s')

    n = len(pairs)
    gm = [sum(p[0][k] for p in pairs) / n for k in range(3)]
    em = [sum(p[1][k] for p in pairs) / n for k in range(3)]
    num = sum((p[0][0]-gm[0])*(p[1][1]-em[1]) - (p[0][1]-gm[1])*(p[1][0]-em[0]) for p in pairs)
    den = sum((p[0][0]-gm[0])*(p[1][0]-em[0]) + (p[0][1]-gm[1])*(p[1][1]-em[1]) for p in pairs)
    yaw = math.atan2(num, den)
    c, s = math.cos(yaw), math.sin(yaw)
    tx = em[0] - (c * gm[0] - s * gm[1])
    ty = em[1] - (s * gm[0] + c * gm[1])
    tz = em[2] - gm[2]
    print(f'yaw = {math.degrees(yaw):+.3f} deg   t = [{tx:+.3f}, {ty:+.3f}, {tz:+.3f}] m')

    resid = []
    for g, e in pairs:
        ax, ay, az = c*g[0]-s*g[1]+tx, s*g[0]+c*g[1]+ty, g[2]+tz
        resid.append(math.dist((ax, ay, az), e))
    resid.sort()
    print(f'residual over the fit window: median {resid[n//2]:.2f} m, max {resid[-1]:.2f} m')

    with open(a.dst, 'w') as f:
        f.write('#timestamp,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz\n')
        for t, x, y, z in gt:
            f.write(f'{int(round(t*1e9))},{c*x-s*y+tx:.6f},{s*x+c*y+ty:.6f},{z+tz:.6f},'
                    '1,0,0,0,0,0,0,0,0,0,0,0,0\n')
    print(f'wrote aligned ground truth -> {a.dst}')


if __name__ == '__main__':
    main()

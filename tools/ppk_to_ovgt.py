#!/usr/bin/env python3
"""
Convert a MUN-FRL RTKLIB .pos PPK solution into the ASL/EuRoC-style CSV that
OpenVINS's `path_gt` launch argument expects.

Output columns are the EuRoC/ASL layout that get_gt_state() expects
(ov_core/src/utils/dataset_reader.h:151-157 maps file col 1,2,3 -> position
and col 4,5,6,7 -> qw,qx,qy,qz):
    timestamp_ns, px, py, pz, qw, qx, qy, qz, vx,vy,vz, bwx,bwy,bwz, bax,bay,baz

The track is resampled to --rate Hz because get_gt_state() only accepts a
match within 0.10 s, while the raw PPK solution is 5 Hz (0.2 s spacing) --
at the raw rate roughly half the lookups fall outside the window and the
ground-truth path silently stays empty.

Positions are local ENU metres with the origin at the first PPK epoch, so the
track starts at (0,0,0) like the OpenVINS estimate does.

Two caveats, both deliberate:
  * The .pos timestamps are GPS time; ROS bag stamps are UTC epoch. GPS ran
    18 leap seconds ahead of UTC in 2022, hence --leap 18. This could not be
    cross-checked against /fix because that topic's real-time solution differs
    from PPK by ~25 m regardless of offset.
  * OpenVINS's global frame is gravity-aligned but its yaw is arbitrary
    (unobservable), so ENU north will not line up with the estimate's x-axis.
    Use --yaw to rotate the ground truth about z until the two overlay.

Usage:
  python3 ppk_to_ovgt.py <in.pos> <out.csv> [--leap 18] [--yaw 0]
"""
import argparse, datetime as dt, math

WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3


def geodetic_to_ecef(lat, lon, h):
    la, lo = math.radians(lat), math.radians(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(la) ** 2)
    return ((N + h) * math.cos(la) * math.cos(lo),
            (N + h) * math.cos(la) * math.sin(lo),
            (N * (1 - WGS84_E2) + h) * math.sin(la))


def ecef_to_enu(x, y, z, lat0, lon0, x0, y0, z0):
    la, lo = math.radians(lat0), math.radians(lon0)
    dx, dy, dz = x - x0, y - y0, z - z0
    return (-math.sin(lo) * dx + math.cos(lo) * dy,
            -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy + math.cos(la) * dz,
            math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy + math.sin(la) * dz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--leap', type=float, default=18.0,
                    help='GPS-UTC leap seconds to subtract (18 for 2022)')
    ap.add_argument('--yaw', type=float, default=0.0,
                    help='degrees to rotate the track about z, to match OpenVINS yaw')
    ap.add_argument('--rate', type=float, default=20.0,
                    help='output Hz (resampled); must beat the 0.10 s lookup window')
    a = ap.parse_args()

    rows = [l.split() for l in open(a.src) if not l.startswith('%') and l.strip()]
    lat0, lon0, h0 = float(rows[0][2]), float(rows[0][3]), float(rows[0][4])
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    c, s_ = math.cos(math.radians(a.yaw)), math.sin(math.radians(a.yaw))

    # build the ENU track at the native PPK epochs first
    track = []
    for r in rows:
        t = dt.datetime.strptime(r[0] + ' ' + r[1], '%Y/%m/%d %H:%M:%S.%f')
        t = t.replace(tzinfo=dt.timezone.utc).timestamp() - a.leap
        e, nn, u = ecef_to_enu(*geodetic_to_ecef(float(r[2]), float(r[3]), float(r[4])),
                               lat0, lon0, x0, y0, z0)
        track.append((t, c * e - s_ * nn, s_ * e + c * nn, u))

    # resample uniformly so every camera frame finds a match inside 0.10 s
    step = 1.0 / a.rate
    t_start, t_end = track[0][0], track[-1][0]
    out, j = [], 0
    tt = t_start
    while tt <= t_end:
        while j + 2 < len(track) and track[j + 1][0] < tt:
            j += 1
        t0, t1 = track[j][0], track[j + 1][0]
        w = 0.0 if t1 == t0 else (tt - t0) / (t1 - t0)
        out.append((tt,) + tuple(track[j][k] + w * (track[j + 1][k] - track[j][k])
                                 for k in (1, 2, 3)))
        tt += step

    n = 0
    with open(a.dst, 'w') as f:
        f.write('#timestamp,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz\n')
        for tt, px, py, pz in out:
            f.write(f'{int(round(tt*1e9))},{px:.6f},{py:.6f},{pz:.6f},'
                    '1,0,0,0,0,0,0,0,0,0,0,0,0\n')
            n += 1

    print(f'wrote {n} poses to {a.dst}')
    print(f'  origin  lat/lon/h = {lat0:.8f}, {lon0:.8f}, {h0:.3f}')
    print(f'  leap    = {a.leap} s   yaw = {a.yaw} deg   rate = {a.rate} Hz')


if __name__ == '__main__':
    main()

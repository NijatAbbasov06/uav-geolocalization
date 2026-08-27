#!/usr/bin/env python3
"""
Turn MUN-FRL camera frames into CAEVL query images.

A map tile covers a fixed patch of ground. A camera frame covers
1.524 * altitude metres (height-limited, for this 49.7x37.3 deg fisheye), so
the two only correspond after the frame is resampled to the tile's ground
extent. Rather than undistorting to a pinhole - which discards the outer ~8 deg
of the fisheye - each output pixel is projected straight onto the ground plane:

    ground (X, Y) at height h  ->  bearing (X/h, Y/h, 1)
                               ->  equidistant distortion
                               ->  source pixel

so the sampling grid depends only on (tile_meters / altitude). Frames too low
to see the whole tile are skipped: you cannot crop an image outward.

Altitude comes from --alt-source. 'gt' uses the PPK solution and is a
diagnostic only - it isolates whether retrieval works. 'vio' uses the OpenVINS
estimate, which is what a real system would have.

Usage:
  python3 make_queries.py <bag_dir> <out_dir> --gt gt_enu.csv
                          [--tile-meters 100] [--every 5] [--alt-source gt|vio]
                          [--vio ov_estimate.txt]
"""
import argparse, json, os, sys
import numpy as np

FX, FY, CX, CY = 829.224, 829.454, 833.937, 562.509
DIST = np.array([-0.0764245, 0.0322856, -0.0445168, 0.0163317])
W, H = 1440, 1080


def equidistant_project(xn, yn):
    """Normalised camera coords -> distorted pixel coords (OpenVINS CamEqui)."""
    r = np.sqrt(xn * xn + yn * yn)
    r = np.maximum(r, 1e-12)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + DIST[0] * th2 + DIST[1] * th2**2
                + DIST[2] * th2**3 + DIST[3] * th2**4)
    s = thd / r
    return FX * (xn * s) + CX, FY * (yn * s) + CY


def ground_grid(tile_m, alt, out_px):
    """Sampling grid mapping an out_px square of ground to source pixels."""
    half = tile_m / 2.0
    g = (np.arange(out_px) + 0.5) / out_px * tile_m - half
    X, Y = np.meshgrid(g, -g)           # image row 0 = north
    return equidistant_project(X / alt, Y / alt)


def load_track(path):
    t, x, y, z = [], [], [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split(',')
        t.append(int(f[0]) / 1e9); x.append(float(f[1]))
        y.append(float(f[2])); z.append(float(f[3]))
    return np.array(t), np.array(x), np.array(y), np.array(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('out')
    ap.add_argument('--gt', required=True, help='gt_enu.csv (bag timebase)')
    ap.add_argument('--tile-meters', type=float, default=100.0)
    ap.add_argument('--out-px', type=int, default=256)
    ap.add_argument('--every', type=int, default=5, help='keep 1 frame in N')
    ap.add_argument('--topic', default='/camera/image_mono')
    ap.add_argument('--alt-source', choices=['gt', 'vio'], default='gt')
    ap.add_argument('--vio', default=None, help='ov_estimate.txt for --alt-source vio')
    ap.add_argument('--min-cover', type=float, default=1.0,
                    help='require camera coverage >= this * tile_meters')
    a = ap.parse_args()

    import cv2
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    gt_t, gt_x, gt_y, gt_z = load_track(a.gt)
    if a.alt_source == 'vio':
        vt, vz = [], []
        for line in open(a.vio):
            if line.startswith('#') or not line.strip():
                continue
            f = line.split()
            if len(f) >= 8:
                vt.append(float(f[0])); vz.append(float(f[7]))
        alt_t, alt_z = np.array(vt), np.array(vz)
    else:
        alt_t, alt_z = gt_t, gt_z

    # height-limited coverage of this camera
    cover_per_m = 2 * np.tan((H / 2) / FY)
    min_alt = a.min_cover * a.tile_meters / cover_per_m
    print(f'tile {a.tile_meters:.0f} m -> needs altitude >= {min_alt:.1f} m '
          f'(coverage = {cover_per_m:.3f} * alt)')

    os.makedirs(a.out, exist_ok=True)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    coords, kept, low, outside, seen = {}, 0, 0, 0, 0

    with Reader(a.bag) as r:
        for conn, _, raw in r.messages():
            if conn.topic != a.topic:
                continue
            seen += 1
            if seen % a.every:
                continue
            m = ts.deserialize_cdr(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if not (gt_t[0] <= t <= gt_t[-1]):
                outside += 1
                continue
            alt = float(np.interp(t, alt_t, alt_z))
            if alt < min_alt:
                low += 1
                continue
            img = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width)
            mx, my = ground_grid(a.tile_meters, alt, a.out_px)
            patch = cv2.remap(img, mx.astype(np.float32), my.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            name = f'q_{t:.6f}.png'.replace('.', '_', 1)
            cv2.imwrite(os.path.join(a.out, name), patch)
            coords[name] = [float(np.interp(t, gt_t, gt_x)),
                            float(np.interp(t, gt_t, gt_y))]
            kept += 1

    with open(os.path.join(a.out, 'coords.json'), 'w') as fh:
        json.dump(coords, fh)
    print(f'\n{seen} frames on {a.topic}, 1 in {a.every} considered')
    print(f'  kept    {kept}')
    print(f'  too low {low}   (altitude < {min_alt:.1f} m)')
    print(f'  outside gt window {outside}')
    print(f'  -> {a.out}')


if __name__ == '__main__':
    main()

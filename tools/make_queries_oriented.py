#!/usr/bin/env python3
"""
Turn MUN-FRL camera frames into CAEVL query images, corrected for attitude.

This replaces make_queries.py, which had two bugs that together made retrieval
hopeless:

1. **No attitude.** Its sampling grid was built in the *camera's* axes, so every
   query came out aligned to wherever the aircraft happened to be pointing,
   while the map tiles are north-up. Over dataset3's 9511 degrees of total
   heading change that is a near-random rotation per frame, and CAEVL - like any
   CNN retrieval - is not rotation invariant. Bank angle was ignored too, and a
   Bell 412 banks hard through turns.

2. **Altitude measured from the wrong datum.** It fed the raw ENU z (relative to
   the first PPK epoch) in as height above ground. The ground is not at z=0 in
   that frame, so every patch covered the wrong ground extent.

Both are fixed here by projecting properly: each output pixel is a point on the
ground plane z = --ground-z, expressed in ENU, rotated into the camera frame
through the full VIO attitude, then pushed through the equidistant model.

    P_enu  = (east, north, ground_z) relative to the camera
    P_imu  = R_GtoI @ P_enu                    (VIO attitude, JPL q_GtoI)
    P_cam  = R_CtoI^T @ (P_imu - p_CinI)       (Kalibr extrinsic)
    pixel  = equidistant(P_cam.xy / P_cam.z)

Attitude necessarily comes from the VIO, since MUN-FRL's PPK solution carries
position only (its quaternion column is an identity placeholder). That is also
the right architecture: VIO supplies attitude, geo-localization supplies
position.

--ground-z is the one number here that is not measured. The dataset's stated
25-170 m AGL against dataset3's z range of -20.7..134.3 puts it near -40 m. It
is worth sweeping: too high and the patch under-covers the tile, too low and it
over-covers, and retrieval degrades either way. Sweeping it and taking the peak
recall is a legitimate calibration, not a fudge.

Usage:
  python3 make_queries_oriented.py <bag_dir> <out_dir> --gt gt_enu.csv \
      --vio ov_estimate.txt [--tile-meters 100] [--every 5] [--ground-z -40]
"""
import argparse, json, os
import numpy as np

# nadir camera, from config/munfrl/kalibr_imucam_chain.yaml
FX, FY, CX, CY = 829.224, 829.454, 833.937, 562.509
DIST = np.array([-0.0764245, 0.0322856, -0.0445168, 0.0163317])
W, H = 1440, 1080

# T_imu_cam: R_CtoI and p_CinI, same file
R_CTOI = np.array([[-0.01039363,  0.99994595,  0.00027614],
                   [-0.99982944, -0.01038820, -0.01527021],
                   [-0.01526652, -0.00043481,  0.99988337]])
P_CINI = np.array([-0.27939175, 0.00394073, -0.00039529])


def equidistant_project(xn, yn):
    """Normalised camera coords -> distorted pixel coords (OpenVINS CamEqui)."""
    r = np.maximum(np.sqrt(xn * xn + yn * yn), 1e-12)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + DIST[0] * th2 + DIST[1] * th2**2
                + DIST[2] * th2**3 + DIST[3] * th2**4)
    s = thd / r
    return FX * (xn * s) + CX, FY * (yn * s) + CY


def quat_2_rot(q):
    """JPL quaternion (x, y, z, w) -> rotation matrix, OpenVINS convention."""
    qv, qw = q[:3], q[3]
    qx = np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])
    return (2 * qw**2 - 1) * np.eye(3) - 2 * qw * qx + 2 * np.outer(qv, qv)


def rotz(th):
    c, sn = np.cos(th), np.sin(th)
    return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])


def estimate_yaw_offset(vio_t, vio_p, gt_t, gt_x, gt_y):
    """Yaw taking OpenVINS's global frame onto the map's ENU frame.

    OpenVINS aligns z to gravity but its heading origin is wherever the filter
    happened to initialise, so R_GtoI is offset from north by an unknown
    constant. Rotating queries by it alone leaves every patch consistently
    mis-rotated, which defeats retrieval just as thoroughly as using no
    attitude at all.

    The offset is recovered by fitting the VIO track onto the ground-truth
    track (the yaw-only case of Umeyama). A real deployment has no ground truth
    to fit against and would need an initial heading from a magnetometer or
    from the first accepted geo-localization fix; here it is an evaluation
    convenience, and the printed residual says whether the fit is trustworthy.
    """
    lo, hi = max(vio_t[0], gt_t[0]), min(vio_t[-1], gt_t[-1])
    m = (vio_t >= lo) & (vio_t <= hi)
    v = vio_p[m][:, :2]
    g = np.column_stack([np.interp(vio_t[m], gt_t, gt_x),
                         np.interp(vio_t[m], gt_t, gt_y)])
    v = v - v.mean(0)
    g = g - g.mean(0)
    th = np.arctan2((v[:, 0] * g[:, 1] - v[:, 1] * g[:, 0]).sum(),
                    (v[:, 0] * g[:, 0] + v[:, 1] * g[:, 1]).sum())
    resid = np.linalg.norm((rotz(th)[:2, :2] @ v.T).T - g, axis=1)
    return th, float(np.sqrt((resid ** 2).mean()))


def load_vio(path):
    """ov_estimate.txt -> (times, quaternions q_GtoI, positions)."""
    t, q, p = [], [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) < 8:
            continue
        t.append(float(f[0]))
        q.append([float(v) for v in f[1:5]])
        p.append([float(v) for v in f[5:8]])
    return np.array(t), np.array(q), np.array(p)


def slerp_at(times, quats, t):
    """Interpolate attitude at time t (nlerp with hemisphere correction)."""
    i = np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2)
    t0, t1 = times[i], times[i + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    q0, q1 = quats[i].copy(), quats[i + 1].copy()
    if q0 @ q1 < 0:                       # take the short way round
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)


def load_track(path):
    t, x, y, z = [], [], [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split(',')
        t.append(int(f[0]) / 1e9); x.append(float(f[1]))
        y.append(float(f[2])); z.append(float(f[3]))
    return np.array(t), np.array(x), np.array(y), np.array(z)


def sample_grid(tile_m, out_px, cam_z, ground_z, R_GtoI):
    """Source pixels for a north-up ground patch under the camera.

    Returns (mx, my, valid). `valid` marks pixels that fall in front of the
    camera and inside the image; a frame is unusable if too few survive.
    """
    half = tile_m / 2.0
    g = (np.arange(out_px) + 0.5) / out_px * tile_m - half
    east, north = np.meshgrid(g, -g)          # row 0 = north
    down = np.full_like(east, ground_z - cam_z)   # negative: ground is below

    P_enu = np.stack([east, north, down], axis=-1)
    P_imu = P_enu @ R_GtoI.T                      # rotate ENU -> IMU
    P_cam = (P_imu - P_CINI) @ R_CTOI             # IMU -> camera (R_CtoI^T)

    z = P_cam[..., 2]
    front = z > 1e-6
    zs = np.where(front, z, 1.0)
    mx, my = equidistant_project(P_cam[..., 0] / zs, P_cam[..., 1] / zs)
    valid = front & (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
    return mx, my, valid


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('out')
    ap.add_argument('--gt', required=True, help='gt_enu.csv (bag timebase)')
    ap.add_argument('--vio', required=True, help='ov_estimate.txt, for attitude')
    ap.add_argument('--tile-meters', type=float, default=100.0)
    ap.add_argument('--out-px', type=int, default=256)
    ap.add_argument('--every', type=int, default=5, help='keep 1 frame in N')
    ap.add_argument('--topic', default='/camera/image_mono')
    ap.add_argument('--ground-z', type=float, default=-40.0,
                    help='ground plane height in the ENU frame; sweep this')
    ap.add_argument('--min-valid', type=float, default=0.98,
                    help='reject a frame unless this fraction of the patch is imaged')
    a = ap.parse_args()

    import cv2
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    gt_t, gt_x, gt_y, gt_z = load_track(a.gt)
    vio_t, vio_q, vio_p = load_vio(a.vio)
    print(f'VIO attitude: {len(vio_t)} poses, {vio_t[0]:.1f}..{vio_t[-1]:.1f}')

    yaw, resid = estimate_yaw_offset(vio_t, vio_p, gt_t, gt_x, gt_y)
    R_align = rotz(yaw)
    print(f'VIO->ENU yaw offset: {np.degrees(yaw):+.2f} deg '
          f'(track-fit residual {resid:.1f} m)')

    os.makedirs(a.out, exist_ok=True)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    coords, kept, partial, outside, seen = {}, 0, 0, 0, 0

    with Reader(a.bag) as r:
        for conn, _, raw in r.messages():
            if conn.topic != a.topic:
                continue
            seen += 1
            if seen % a.every:
                continue
            m = ts.deserialize_cdr(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if not (gt_t[0] <= t <= gt_t[-1] and vio_t[0] <= t <= vio_t[-1]):
                outside += 1
                continue

            # R_GtoI is relative to OpenVINS's arbitrary heading origin;
            # compose with R_align to express attitude relative to map north.
            R_GtoI = quat_2_rot(slerp_at(vio_t, vio_q, t)) @ R_align.T
            cam_z = float(np.interp(t, gt_t, gt_z))
            mx, my, valid = sample_grid(a.tile_meters, a.out_px,
                                        cam_z, a.ground_z, R_GtoI)
            if valid.mean() < a.min_valid:
                partial += 1
                continue

            img = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width)
            patch = cv2.remap(img, mx.astype(np.float32), my.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            name = f'q_{t:.6f}.png'.replace('.', '_', 1)
            cv2.imwrite(os.path.join(a.out, name), patch)
            coords[name] = [float(np.interp(t, gt_t, gt_x)),
                            float(np.interp(t, gt_t, gt_y))]
            kept += 1

    with open(os.path.join(a.out, 'coords.json'), 'w') as fh:
        json.dump(coords, fh)
    import pickle
    with open(os.path.join(a.out, 'coords.pkl'), 'wb') as fh:
        pickle.dump({os.path.splitext(k)[0]: tuple(v) for k, v in coords.items()}, fh)

    print(f'\n{seen} frames on {a.topic}, 1 in {a.every} considered')
    print(f'  kept              {kept}')
    print(f'  patch not covered {partial}   (raise --ground-z or lower --tile-meters)')
    print(f'  outside window    {outside}')
    print(f'  -> {a.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Rewrite a converted MUN-FRL rosbag2 with corrected IMU timestamps.

Why: /imu/data carries driver *receipt* times (bursty, ~19 ms jitter), not
sampling times. The true 400 Hz hardware clock is on /imu/time_ref.time_ref,
which shows an exact 2500.0 us period. Feeding the receipt times to OpenVINS
makes every preintegration interval wrong and the filter diverges quickly.

How: /imu/data and /imu/time_ref share identical header stamps, giving an exact
1:1 pairing. We fit  ros_time = a * hw_time + b  by least squares over the whole
sequence (a absorbs the -49.9 us/s clock drift, b the mean latency) and restamp
each IMU sample to a*hw + b. Residual constant latency between camera and IMU is
left to OpenVINS's calib_cam_timeoffset.

Camera stamps are deliberately NOT touched -- see the module docstring note in
the project log: /imu/time_ref_cam is a 20 Hz pulse while the camera free-runs
at 15 Hz, so the two are not in 1:1 correspondence.

Usage:
  python3 restamp_imu.py <src_bag_dir> <dst_bag_dir> [topic ...]
"""
import sys, heapq
import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

DEFAULT_TOPICS = ['/imu/data', '/camera/image_mono', '/fix']
IMU_TOPIC, REF_TOPIC = '/imu/data', '/imu/time_ref'
BUFFER = 5000  # messages held for reordering; max restamp shift is ~20 ms


def stamp_s(s):
    return s.sec + s.nanosec * 1e-9


def main():
    src, dst = sys.argv[1], sys.argv[2]
    topics = sys.argv[3:] or DEFAULT_TOPICS
    ts = get_typestore(Stores.ROS2_HUMBLE)

    # --- pass 1: fit the hardware -> ROS clock model -----------------------
    hw, ros = [], []
    with Reader(src) as r:
        for conn, _, raw in r.messages():
            if conn.topic != REF_TOPIC:
                continue
            m = ts.deserialize_cdr(raw, conn.msgtype)
            hw.append(stamp_s(m.time_ref))
            ros.append(stamp_s(m.header.stamp))
    hw, ros = np.array(hw), np.array(ros)
    a, b = np.polyfit(hw, ros, 1)
    resid = ros - (a * hw + b)
    print(f'clock fit: a={a:.9f}  b={b:.6f}  '
          f'residual std={resid.std()*1e3:.3f} ms over {len(hw)} samples')

    # map each IMU header stamp -> corrected stamp, keyed by the shared header
    key2new = {}
    with Reader(src) as r:
        for conn, _, raw in r.messages():
            if conn.topic != REF_TOPIC:
                continue
            m = ts.deserialize_cdr(raw, conn.msgtype)
            key2new[round(stamp_s(m.header.stamp), 9)] = a * stamp_s(m.time_ref) + b

    # --- pass 2: stream, restamp, write in timestamp order -----------------
    written = missed = 0
    # rosbags >= 0.10 requires the rosbag2 format version explicitly;
    # 9 is what the converted MUN-FRL bags already use.
    with Reader(src) as r, Writer(dst, version=9) as w:
        outconn = {}
        for c in r.connections:
            if c.topic in topics:
                outconn[c.topic] = w.add_connection(c.topic, c.msgtype, typestore=ts)

        buf, seq = [], 0
        def flush(force=False):
            nonlocal written
            while buf and (force or len(buf) > BUFFER):
                t, _, tp, data = heapq.heappop(buf)
                w.write(outconn[tp], t, data)
                written += 1

        for conn, t, raw in r.messages():
            if conn.topic not in outconn:
                continue
            if conn.topic == IMU_TOPIC:
                m = ts.deserialize_cdr(raw, conn.msgtype)
                new = key2new.get(round(stamp_s(m.header.stamp), 9))
                if new is None:
                    missed += 1
                    continue
                m.header.stamp.sec = int(new)
                m.header.stamp.nanosec = int(round((new - int(new)) * 1e9))
                raw = ts.serialize_cdr(m, conn.msgtype)
                t = int(new * 1e9)
            heapq.heappush(buf, (t, seq, conn.topic, raw))
            seq += 1
            flush()
        flush(force=True)

    print(f'wrote {written} messages to {dst}')
    print(f'IMU samples with no matching /imu/time_ref (dropped): {missed}')


if __name__ == '__main__':
    main()

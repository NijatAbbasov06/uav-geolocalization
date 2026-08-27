#!/usr/bin/env python3
"""
Download clean satellite imagery for a MUN-FRL flight and stitch it into one
georeferenced map.

Why not use the map MUN-FRL ships: the authors painted the flight path onto it
in green, 18.6 m thick on the ground. Tiles are cut *along* that path, so the
painted band lands in the middle of every tile a query could correctly match to.
Erasing it by inpainting leaves a smear across 23% of each tile - better than a
false edge, but still a quarter of the matching signal gone. Their map is also
0.79 m/px, blurrier than the 0.39 m/px queries, so tiles get upscaled.

Where the imagery comes from: Esri's World Imagery basemap, which is free, needs
no API key, and permits this use with attribution. Google and Bing tiles require
a paid key and their terms forbid bulk download, so they are not an option here.

How we know which imagery to fetch: the PPK solution stores latitude and
longitude, so the flight's footprint on the Earth is known exactly. Web basemaps
are pre-cut into 256x256 tiles indexed by (zoom, x, y) in Web Mercator, and a
bounding box maps straight onto a rectangle of those indices.

Zoom 19 gives ~0.21 m/px here, sharper than the queries, so tiles are downsampled
into place rather than upscaled.

One thing this does NOT fix, and should not: the imagery dates from whenever Esri
last flew the area, not 25 May 2022. Crops, construction and season will differ.
That is a real domain gap the method has to survive in deployment - unlike a
painted line, it belongs in the results.

Outputs <out>.jpg plus <out>.json, the sidecar holding the ENU->pixel transform
for munfrl_map_tiles.py --georef.

Usage:
  python3 fetch_basemap.py --pos bell412_dataset3_ppk.pos \
      --gt gt_enu_tileframe.csv --out maps/dataset3_esri --zoom 19
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
from io import BytesIO

import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

URL = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
       'World_Imagery/MapServer/tile/{z}/{y}/{x}')
ATTRIBUTION = ('Esri, Maxar, Earthstar Geographics, and the GIS User Community')


def lonlat_to_pixel(lon, lat, z):
    """Web Mercator pixel coordinates at zoom z (256 px tiles).

    numpy rather than math so it takes whole trajectories as well as scalars.
    """
    n = 256 * 2 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    s = np.sin(np.radians(np.asarray(lat)))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return x, y


def read_pos(path):
    """RTKLIB .pos -> (time, lat, lon).

    Time is GPST seconds on the Unix epoch. GPST runs ahead of UTC by the
    accumulated leap seconds (18 s in 2022); main() removes that by aligning
    the two files' first epochs rather than hard-coding the value.
    """
    import datetime as dt
    t, lat, lon = [], [], []
    for line in open(path):
        if line.startswith('%') or not line.strip():
            continue
        f = line.split()
        stamp = dt.datetime.strptime(f[0] + ' ' + f[1], '%Y/%m/%d %H:%M:%S.%f')
        t.append(stamp.replace(tzinfo=dt.timezone.utc).timestamp())
        lat.append(float(f[2])); lon.append(float(f[3]))
    return np.array(t), np.array(lat), np.array(lon)


def read_enu(path):
    """gt_enu.csv -> (time, east, north). Time is Unix seconds."""
    t, e, n = [], [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split(',')
        t.append(int(f[0]) / 1e9)
        e.append(float(f[1])); n.append(float(f[2]))
    return np.array(t), np.array(e), np.array(n)


def solve_enu_to_pixel(lat, lon, east, north, z):
    """Least-squares map from ENU metres to Web Mercator pixels.

    Rather than assume how ppk_to_ovgt.py defined its ENU frame, this fits the
    relation directly from the two representations of the same trajectory. Over
    a 2 km site Mercator scale is effectively constant, so an affine fit is
    exact to well under a pixel - and the reported residual proves it, instead
    of us trusting a convention that has already been wrong once today.
    """
    px, py = lonlat_to_pixel(lon, lat, z)
    A = np.column_stack([east, north, np.ones_like(east)])
    cx, *_ = np.linalg.lstsq(A, px, rcond=None)
    cy, *_ = np.linalg.lstsq(A, py, rcond=None)
    res = np.hypot(A @ cx - px, A @ cy - py)
    return cx, cy, float(res.max())


def fetch(session, z, x, y, cache):
    p = os.path.join(cache, f'{z}_{x}_{y}.jpg')
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return x, y, p
    for attempt in range(4):
        try:
            r = session.get(URL.format(z=z, x=x, y=y), timeout=30)
            if r.status_code == 200 and r.content:
                with open(p, 'wb') as fh:
                    fh.write(r.content)
                return x, y, p
        except requests.RequestException:
            pass
    return x, y, None


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--pos', required=True, help='RTKLIB .pos with lat/lon')
    ap.add_argument('--gt', required=True, help='gt_enu.csv, the tile frame')
    ap.add_argument('--out', required=True, help='output path without extension')
    ap.add_argument('--zoom', type=int, default=19)
    ap.add_argument('--margin', type=float, default=400.0,
                    help='metres of imagery beyond the flight bounding box')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    t_pos, lat, lon = read_pos(a.pos)
    t_gt, e_gt, n_gt = read_enu(a.gt)

    # The .pos is 5 Hz GPST; gt_enu.csv is resampled to 20 Hz UTC. Align the
    # two timebases on their first epoch - the difference should come out as
    # the GPS-UTC leap second count (18 s in 2022), which is a good check that
    # these really are the same flight rather than a coincidence of duration.
    shift = t_pos[0] - t_gt[0]
    dur_pos, dur_gt = t_pos[-1] - t_pos[0], t_gt[-1] - t_gt[0]
    print(f'time alignment: .pos leads gt by {shift:.1f} s '
          f'(expect ~18 s of leap seconds)')
    print(f'  durations {dur_pos:.1f} s vs {dur_gt:.1f} s')
    if abs(dur_pos - dur_gt) > 2.0:
        sys.exit('the two files cover different intervals')

    tq = t_pos - shift
    inside = (tq >= t_gt[0]) & (tq <= t_gt[-1])
    tq, lat, lon = tq[inside], lat[inside], lon[inside]
    east = np.interp(tq, t_gt, e_gt)
    north = np.interp(tq, t_gt, n_gt)
    print(f'  matched {len(tq)} epochs')

    cx, cy, res = solve_enu_to_pixel(lat, lon, east, north, a.zoom)
    mpp = 156543.03392 * math.cos(math.radians(lat.mean())) / 2 ** a.zoom
    print(f'zoom {a.zoom}: {mpp:.4f} m/px')
    print(f'ENU->pixel fit residual: {res:.2f} px ({res * mpp:.2f} m)')
    if res > 2.0:
        sys.exit('fit is too loose - the .pos and gt files may not correspond')

    # bounding box in pixels, padded, snapped out to whole tiles
    m = a.margin / mpp
    ex = np.array([east.min() - a.margin, east.max() + a.margin])
    nx_ = np.array([north.min() - a.margin, north.max() + a.margin])
    corners = [(e, n) for e in ex for n in nx_]
    P = np.array([[cx[0] * e + cx[1] * n + cx[2],
                   cy[0] * e + cy[1] * n + cy[2]] for e, n in corners])
    px0, py0 = P[:, 0].min(), P[:, 1].min()
    px1, py1 = P[:, 0].max(), P[:, 1].max()

    tx0, ty0 = int(px0 // 256), int(py0 // 256)
    tx1, ty1 = int(px1 // 256), int(py1 // 256)
    cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
    total = cols * rows
    print(f'tiles: {cols} x {rows} = {total}   '
          f'stitched {cols * 256} x {rows * 256} px')

    cache = a.out + '_tiles'
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)

    session = requests.Session()
    session.headers['User-Agent'] = 'munfrl-research/1.0'
    jobs = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]

    canvas = Image.new('RGB', (cols * 256, rows * 256))
    done = failed = 0
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, unit='tile')
    except ImportError:
        bar = None

    with cf.ThreadPoolExecutor(a.workers) as ex_:
        for x, y, p in ex_.map(lambda j: fetch(session, a.zoom, j[0], j[1], cache),
                               jobs):
            if p is None:
                failed += 1
            else:
                canvas.paste(Image.open(p).convert('RGB'),
                             ((x - tx0) * 256, (y - ty0) * 256))
                done += 1
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    print(f'fetched {done}/{total} tiles ({failed} failed)')
    if failed:
        print('  re-run to retry the failures; completed tiles are cached')

    # Reduce the affine to the scale+offset form munfrl_map_tiles.py uses:
    #     px =  x / mpp + ox      py = -y / mpp + oy
    # Web Mercator is north-up, so the cross terms (how much east displaces a
    # pixel vertically, and vice versa) should vanish. Check rather than assume.
    ox = cx[2] - tx0 * 256
    oy = cy[2] - ty0 * 256
    span = max(np.ptp(east), np.ptp(north))
    cross = max(abs(cx[1]), abs(cy[0])) * span
    mpp_x, mpp_y = 1.0 / cx[0], -1.0 / cy[1]
    print(f'  scale from east {mpp_x:.5f} m/px, from north {mpp_y:.5f} m/px '
          f'(differ by {abs(mpp_x - mpp_y) * 1000:.2f} mm/px)')
    print(f'  cross-term over the whole site: {cross:.2f} px '
          f'({cross * mpp:.2f} m)')
    if cross > 3.0:
        print('  WARNING: not north-up enough for the simple form; '
              'the tiler would be slightly rotated')
    mpp_fit = 0.5 * (mpp_x + mpp_y)

    georef = {
        'm_per_px': mpp_fit,
        'ox': ox,
        'oy': oy,
        'note': 'px = x / m_per_px + ox ; py = -y / m_per_px + oy',
        'zoom': a.zoom,
        'size': [canvas.size[0], canvas.size[1]],
        'fit_residual_px': res,
        'cross_term_px': cross,
        'source': 'Esri World Imagery',
        'attribution': ATTRIBUTION,
    }
    canvas.save(a.out + '.jpg', quality=92)
    with open(a.out + '.json', 'w') as fh:
        json.dump(georef, fh, indent=2)
    print(f'\nwrote {a.out}.jpg  ({canvas.size[0]}x{canvas.size[1]})')
    print(f'wrote {a.out}.json')
    print(f'\nattribution required when publishing: {ATTRIBUTION}')


if __name__ == '__main__':
    main()

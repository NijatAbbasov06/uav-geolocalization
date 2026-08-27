#!/usr/bin/env python3
"""
Cut a MUN-FRL reference map into a CAEVL tile database.

MUN-FRL ships one JPEG per sequence with the georeferencing baked into the
filename rather than into the file, e.g.

    dataset32_1_26_pxpm_map_parameters_0_793651_-1788_-1465.jpg
                                       ^^^^^^^^ ^^^^^ ^^^^^
                                       m/px     off_x off_y

which maps the sequence's local ENU frame (origin = first PPK epoch, the same
frame ppk_to_ovgt.py emits) to map pixels as

    px =  x / mpp - off_x
    py = -off_y - y / mpp          (y flips: ENU north is up, pixels count down)

Verified against dataset3: 6141/6161 PPK track points land inside the image and
the southern extreme falls at py=2991.4 against an image height of 2991, i.e.
the map was cropped to the flight bounds.

Output is what `localizer.py build` wants: a folder of PNG tiles plus a JSON
{tile_name: [x, y]} of tile-centre coordinates in that same ENU frame.

Usage:
  python3 munfrl_map_tiles.py <map.jpg> <out_dir> [--tile-meters 200]
                              [--spacing 50] [--track gt_enu.csv] [--corridor 400]
"""
import argparse, json, os, re, sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def parse_map_params(path):
    """Pull (m_per_px, off_x, off_y) out of the MUN-FRL filename."""
    name = os.path.basename(path)
    m = re.search(r'_(\d+)_(\d+)_(-?\d+)_(-?\d+)\.jpg$', name, re.I)
    if not m:
        raise SystemExit(f'cannot parse map parameters from {name!r}')
    mpp = float(f'{m.group(1)}.{m.group(2)}')
    return mpp, int(m.group(3)), int(m.group(4))


def calibrate_from_drawn_track(map_path, track_xy):
    """Recover the map georeferencing by fitting the flight track onto the
    green curve MUN-FRL drew on the map.

    The parameters in the filename are not reliable. For dataset3 the scale is
    right (~1.26 px/m) but the offsets are wrong by 378 m east and 256 m north,
    putting every tile ~456 m from the ground it claims to show. A bounds check
    cannot catch this - the track still lands inside the image - yet it quietly
    destroys retrieval, because each query is then scored against tiles of a
    different field entirely.

    The authors drew the true path on each map in green, so the transform can be
    solved for directly: slide and scale the known ENU track until it lies on
    that curve. The returned residual is the check; a good fit lands well inside
    the ~8 px line width.

    Returns (m_per_px, ox, oy, residual_px) for
        px = x / mpp + ox
        py = -y / mpp + oy
    or None if no drawn track could be found.
    """
    import cv2
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    im = np.array(Image.open(map_path).convert('RGB')).astype(int)
    sat = im.max(2) - im.min(2)
    green = ((sat > 30) & (im[..., 1] > im[..., 0])
             & (im[..., 1] > im[..., 2])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    if n < 2:
        return None
    # the largest blob is the track; the rest are Google Earth's UI pins
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == k).astype(np.uint8)
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    H, W = mask.shape

    x, y = track_xy[:, 0], track_xy[:, 1]
    ys, xs = np.nonzero(mask)

    def cost(sc, ox, oy):
        px, py = sc * x + ox, -sc * y + oy
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if ok.mean() < 0.99:
            return 1e9
        return float(dist[py[ok].astype(int), px[ok].astype(int)].mean())

    # seed from the bounding boxes, which pins the scale to about 1%
    sc = 0.5 * ((xs.max() - xs.min()) / np.ptp(x)
                + (ys.max() - ys.min()) / np.ptp(y))
    ox = xs.min() - sc * x.min()
    oy = ys.min() + sc * y.max()

    best = (cost(sc, ox, oy), (sc, ox, oy))
    ds, dx, dy = 0.004, 24.0, 24.0
    for _ in range(7):
        sc, ox, oy = best[1]
        for S in (sc - ds, sc, sc + ds):
            for OX in (ox - dx, ox, ox + dx):
                for OY in (oy - dy, oy, oy + dy):
                    c = cost(S, OX, OY)
                    if c < best[0]:
                        best = (c, (S, OX, OY))
        ds, dx, dy = ds / 2.4, dx / 2.4, dy / 2.4

    sc, ox, oy = best[1]
    return 1.0 / sc, ox, oy, best[0]


def remove_drawn_overlays(img, feather=3, radius=7):
    """Paint out the annotations MUN-FRL drew on the map.

    Each map carries the true flight path drawn over it in green, plus a Google
    Earth pin or two. None of that is ground the camera ever saw, and because
    tiles are cut *along* the flight path, the painted band lands in nearly
    every tile a query could match to - a strong Canny edge present in all of
    them and in none of the queries. Left in, it is worse than missing data: it
    is a consistent false signal.

    The separation is easy because the satellite imagery is monochrome and only
    the annotations carry colour, so thresholding on saturation catches every
    drawn pixel and no real terrain.

    Inpainting cannot recover what the paint covered - expect a smooth smear
    roughly the width of the line where it crossed textured ground. That is the
    intended trade: a weak edge that loses information, rather than a strong one
    that invents it.

    Returns (image, pixels_repainted).
    """
    import cv2
    import numpy as np

    a = np.array(img)
    sat = a.max(2).astype(np.int16) - a.min(2).astype(np.int16)
    mask = (sat > 30).astype(np.uint8)
    if mask.sum() == 0:
        return img, 0
    k = np.ones((2 * feather + 1, 2 * feather + 1), np.uint8)
    mask = cv2.dilate(mask, k)          # cover the anti-aliased fringe too
    out = cv2.inpaint(a, mask, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(out), int(mask.sum())


def enu_to_px(x, y, mpp, ox, oy):
    return x / mpp + ox, -y / mpp + oy


def load_track(path):
    pts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f = line.split(',')
            pts.append((float(f[1]), float(f[2])))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map'); ap.add_argument('out')
    ap.add_argument('--tile-meters', type=float, default=200.0,
                    help='ground size of one tile; match the camera footprint')
    ap.add_argument('--spacing', type=float, default=50.0,
                    help='distance between tile centres; overlap helps recall')
    ap.add_argument('--tile-px', type=int, default=256,
                    help='output tile size in pixels')
    ap.add_argument('--track', default=None,
                    help='gt_enu.csv - keep only tiles near the flight path')
    ap.add_argument('--corridor', type=float, default=400.0,
                    help='metres either side of the track to keep')
    ap.add_argument('--georef', default=None,
                    help='sidecar .json from fetch_basemap.py; skips both the '
                         'filename parsing and the drawn-track calibration')
    a = ap.parse_args()

    if a.georef:
        with open(a.georef) as fh:
            gr = json.load(fh)
        mpp, ox, oy = gr['m_per_px'], gr['ox'], gr['oy']
        print(f'georef from {a.georef}: {mpp:.6f} m/px, offsets '
              f'({ox:.1f}, {oy:.1f})')
        print(f'   source: {gr.get("source", "?")}')
    else:
        mpp, ox, oy = parse_map_params(a.map)
    img = Image.open(a.map).convert('RGB')
    W, H = img.size
    track = load_track(a.track) if a.track else None
    if track:
        print(f'   track: {len(track)} points, corridor {a.corridor:.0f} m')
        import numpy as _np
        fit = None if a.georef else calibrate_from_drawn_track(
            a.map, _np.array(track))
        if fit is None:
            if not a.georef:
                print('   no drawn track found on the map; '
                      'keeping the filename values')
        else:
            mpp_f, ox_f, oy_f, resid = fit
            print(f'   filename : {mpp:.6f} m/px, offsets ({ox}, {oy})')
            print(f'   track fit: {mpp_f:.6f} m/px, offsets '
                  f'({ox_f:.1f}, {oy_f:.1f}), residual {resid:.2f} px')
            if resid < 6.0:
                mpp, ox, oy = mpp_f, ox_f, oy_f
                print('   -> using the fitted transform')
            else:
                print('   -> FIT IS POOR, keeping filename values; check this')
    elif not a.georef:
        # without a track there is nothing to calibrate against, and the
        # filename convention has already proved wrong at least once
        ox, oy = -ox, -oy
        print('   WARNING: no --track given, so the georeferencing is '
              'unverified')

    print(f'map {W}x{H} px, {mpp:.6f} m/px, offsets ({ox}, {oy})')
    print(f'   covers {W*mpp:.0f} x {H*mpp:.0f} m')

    # must happen after the calibration above, which needs the drawn track
    # fetched basemaps have nothing painted on them
    img, repainted = (img, 0) if a.georef else remove_drawn_overlays(img)
    print(f'   repainted {repainted} annotation px '
          f'({100.0 * repainted / (W * H):.2f}% of the map)')

    half = a.tile_meters / 2.0
    # walk the ENU rectangle the image spans
    x_lo, y_hi = (0 - ox) * mpp, (oy - 0) * mpp
    x_hi, y_lo = (W - ox) * mpp, (oy - H) * mpp

    os.makedirs(a.out, exist_ok=True)
    coords, n, skipped = {}, 0, 0
    y = y_lo + half
    while y <= y_hi - half:
        x = x_lo + half
        while x <= x_hi - half:
            if track:
                near = any((px - x) ** 2 + (py - y) ** 2 <= a.corridor ** 2
                           for px, py in track)
                if not near:
                    skipped += 1
                    x += a.spacing
                    continue
            cx, cy = enu_to_px(x, y, mpp, ox, oy)
            box = (int(round(cx - half / mpp)), int(round(cy - half / mpp)),
                   int(round(cx + half / mpp)), int(round(cy + half / mpp)))
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                x += a.spacing
                continue
            name = f'tile_{n:06d}.png'
            img.crop(box).resize((a.tile_px, a.tile_px), Image.BILINEAR).save(
                os.path.join(a.out, name))
            coords[name] = [round(x, 3), round(y, 3)]
            n += 1
            x += a.spacing
        y += a.spacing

    coords_path = os.path.join(a.out, 'coords.json')
    with open(coords_path, 'w') as fh:
        json.dump(coords, fh)
    print(f'\nwrote {n} tiles to {a.out}')
    if track:
        print(f'   ({skipped} grid positions skipped as off-track)')
    print(f'   coords -> {coords_path}')
    print(f'   tile = {a.tile_meters:.0f} m ground -> {a.tile_px} px '
          f'({a.tile_meters/a.tile_px:.3f} m/px)')


if __name__ == '__main__':
    main()

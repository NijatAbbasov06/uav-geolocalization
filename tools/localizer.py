#!/usr/bin/env python
"""Runtime CPU geo-localizer: one UAV frame in, one position out.

Split by what is knowable in advance:

    OFFLINE (once, on the ground)   encode the map tiles into descriptors
    RUNTIME (per frame, in flight)  encode the incoming UAV image, search the map

The map is known before takeoff, so encoding it is a build step. Incoming UAV
frames are not, so those are encoded live. Nothing about a frame is precomputed.

Measured on this machine (20 threads, CPU only):
    encode one 256x256 frame   ~62 ms
    search 14,265 descriptors  ~2.6 ms
    -> ~15 Hz

Build the map descriptors once
------------------------------
    python localizer.py build \
        --weights caevl/models/trained/ft/stagetwo_vild/stagetwo.pth \
        --database_folder data/UAV/caevl/train \
        --database_coords data/UAV/caevl/train_coords_eval.pkl \
        --out alto_train_map.npz

Then localize a live stream of frames
-------------------------------------
    python localizer.py stream \
        --map alto_train_map.npz \
        --queries_folder data/UAV/Train/query_images \
        --fps 20 --out fixes.csv

Or use it from your own code / a ROS node
-----------------------------------------
    loc = Localizer('alto_train_map.npz')
    fix = loc.localize(pil_image)     # {'x':.., 'y':.., 'score':.., 'margin':.., 'ms':..}
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from caevl.ae.tools.load_ae import load_ae
from caevl.ft_stage.model.encoder import CaevlFT
from caevl.evaluation.models import load_weights
import caevl.data_generation.dataset.augmentations as aug
import torchvision.transforms as transforms


def build_encoder(weights_path, device='cpu', image_size=(256, 256)):
    """Rebuild the stage-two encoder and load its weights.

    The architecture is reconstructed from backbone_config.yml, which stage two
    saved next to the checkpoint. Only the encoder survives - the projectors and
    decoder exist during training and are discarded here.
    """
    config_path = os.path.join(os.path.dirname(weights_path), 'backbone_config.yml')
    autoencoder = load_ae(config_path)

    model = CaevlFT(backbone=autoencoder.encoder, projector=None, local_projector=None,
                    device=device, invariance_coeff=0, std_coeff=0, cov_coeff=0, alpha=0)
    # CaevlFT records the device but does not move its parameters, so do it here
    # - before the dummy pass below, or the input and weights end up on
    # different devices.
    model = model.to(device)

    # The bottleneck is a LazyLinear: it has no weights until a first forward
    # pass infers its input size, so load_state_dict would fail without this.
    with torch.no_grad():
        model.backbone(torch.zeros(1, 1, *image_size, device=device))

    load_weights(model, weights_path, device=device, strict=False)
    return model.backbone.eval().to(device)


def build_transform(image_size=(256, 256)):
    """The exact preprocessing eval.py applies: resize, grey, Canny, tensor."""
    return transforms.Compose([
        aug.Resize(image_size, antialias=True),
        transforms.Grayscale(),
        aug.CannyMask(low_threshold=70, high_threshold=200, apertureSize=3, L2gradient=False),
        transforms.ToTensor(),
    ])


class Localizer:
    """Loads the model and the prebuilt map once; localizes frames one at a time."""

    def __init__(self, map_path, threads=None, device='cpu'):
        if threads:
            torch.set_num_threads(threads)

        blob = np.load(map_path, allow_pickle=True)
        self.descriptors = blob['descriptors']          # (n, 1024), L2-normalised
        self.coords = blob['coords']                    # (n, 2)
        self.names = blob['names']
        weights_path = str(blob['weights'])
        self.image_size = tuple(int(v) for v in blob['image_size'])

        self.device = device
        self.encoder = build_encoder(weights_path, device=device, image_size=self.image_size)
        self.transform = build_transform(self.image_size)

        print(f'map: {len(self.coords)} tiles | encoder on {device} | '
              f'{torch.get_num_threads()} threads')

    def encode(self, image):
        """PIL image -> unit-norm 1024-d descriptor."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            descriptor = self.encoder(tensor)[0].cpu().numpy()
        return descriptor / (np.linalg.norm(descriptor) + 1e-12)

    def localize(self, image, topk=5):
        """One frame in, one position fix out.

        Returns
        -------
        dict with
            x, y     : position of the best-matching map tile
            score    : cosine similarity of that match, in [-1, 1]
            margin   : score gap to the runner-up. A confident match beats its
                       runner-up clearly; an ambiguous one does not. This is the
                       more useful confidence signal of the two.
            ms       : wall-clock milliseconds for encode + search
            topk_*   : the ranked shortlist, for multi-hypothesis use
        """
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)

        start = time.perf_counter()
        query = self.encode(image)
        scores = self.descriptors @ query
        ranked = np.argpartition(-scores, topk)[:topk]
        ranked = ranked[np.argsort(-scores[ranked])]
        elapsed = (time.perf_counter() - start) * 1000

        best = ranked[0]
        return {
            'x': float(self.coords[best][0]),
            'y': float(self.coords[best][1]),
            'score': float(scores[best]),
            'margin': float(scores[best] - scores[ranked[1]]) if topk > 1 else 0.0,
            'ms': elapsed,
            'topk_idx': ranked.tolist(),
            'topk_xy': self.coords[ranked].tolist(),
            'topk_score': scores[ranked].tolist(),
        }


def cmd_build(args):
    """Encode every map tile once. This is the ground-station step."""
    from caevl.evaluation.test_dataset import read_images_paths, load_dict, paths_to_coords

    paths = read_images_paths(args.database_folder)
    coords = np.array(paths_to_coords(paths, load_dict(args.database_coords)), dtype=np.float64)

    if args.threads:
        torch.set_num_threads(args.threads)
    image_size = tuple(args.image_size)
    encoder = build_encoder(args.weights, device=args.device, image_size=image_size)
    transform = build_transform(image_size)

    descriptors = np.zeros((len(paths), 1024), dtype=np.float32)
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            batch = torch.stack([transform(Image.open(p)) for p in chunk]).to(args.device)
            out = encoder(batch).cpu().numpy()
            descriptors[i:i + len(chunk)] = out
            if i % (args.batch_size * 20) == 0:
                done = i + len(chunk)
                print(f'  {done}/{len(paths)}  ({done / max(time.time() - start, 1e-9):.0f} img/s)')

    descriptors /= (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-12)
    np.savez_compressed(args.out,
                        descriptors=descriptors, coords=coords,
                        names=np.array([os.path.basename(p) for p in paths]),
                        weights=os.path.abspath(args.weights),
                        image_size=np.array(image_size))
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, '
          f'{len(paths)} tiles, {time.time() - start:.0f}s)')


def cmd_stream(args):
    """Feed frames through one at a time, as they would arrive in flight."""
    from caevl.evaluation.test_dataset import read_images_paths

    localizer = Localizer(args.map, threads=args.threads, device=args.device)
    frames = read_images_paths(args.queries_folder)
    if args.limit:
        frames = frames[:args.limit]

    truth = None
    if args.queries_coords:
        # load_dict handles .json as well as .pkl; cmd_build already uses it,
        # and the tile scripts emit both, so accept either here too.
        from caevl.evaluation.test_dataset import load_dict
        truth = load_dict(args.queries_coords)

    handle_out = open(args.out, 'w') if args.out else None
    if handle_out:
        handle_out.write('frame,timestamp,name,pred_x,pred_y,score,margin,ms')
        handle_out.write(',gt_x,gt_y,error_m\n' if truth else '\n')

    timings, errors = [], []
    for i, path in enumerate(frames):
        fix = localizer.localize(path)
        timestamp = i / args.fps
        timings.append(fix['ms'])

        row = (f"{i},{timestamp:.4f},{os.path.basename(path)},"
               f"{fix['x']:.2f},{fix['y']:.2f},{fix['score']:.4f},"
               f"{fix['margin']:.4f},{fix['ms']:.1f}")

        if truth is not None:
            name = os.path.basename(path)
            gt = truth.get(name, truth.get(os.path.splitext(name)[0]))
            if gt is not None:
                error = float(np.hypot(fix['x'] - gt[0], fix['y'] - gt[1]))
                errors.append(error)
                row += f',{gt[0]:.2f},{gt[1]:.2f},{error:.2f}'
            else:
                row += ',,,'

        if handle_out:
            handle_out.write(row + '\n')
        if i % args.print_every == 0:
            extra = f'  err {errors[-1]:7.1f} m' if errors else ''
            print(f'[{timestamp:7.2f}s] frame {i:6d}  '
                  f'({fix["x"]:.0f}, {fix["y"]:.0f})  '
                  f'score {fix["score"]:.3f}  {fix["ms"]:5.1f} ms{extra}')

    if handle_out:
        handle_out.close()

    timings = np.array(timings)
    print(f'\n{len(frames)} frames localized live')
    print(f'  per frame : {timings.mean():.1f} ms mean, '
          f'{np.percentile(timings, 95):.1f} ms p95  ->  {1000 / timings.mean():.1f} Hz')
    if errors:
        errors = np.array(errors)
        print(f'  error     : median {np.median(errors):.1f} m, '
              f'{100 * (errors <= 50).mean():.1f}% within 50 m, '
              f'{100 * (errors <= 100).mean():.1f}% within 100 m')
    if args.out:
        print(f'  wrote {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='encode the map tiles once (offline)')
    b.add_argument('--weights', required=True)
    b.add_argument('--database_folder', required=True)
    b.add_argument('--database_coords', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--image_size', type=int, nargs=2, default=[256, 256])
    b.add_argument('--batch_size', type=int, default=64)
    b.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='the build step may use the GPU; runtime does not')
    b.add_argument('--threads', type=int, default=None)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser('stream', help='localize frames one at a time (runtime)')
    s.add_argument('--map', required=True)
    s.add_argument('--queries_folder', required=True)
    s.add_argument('--queries_coords', default=None,
                   help='optional ground truth, to report live error')
    s.add_argument('--out', default=None, help='CSV of fixes, feeds the fusion step')
    s.add_argument('--fps', type=float, default=20.0)
    s.add_argument('--limit', type=int, default=None)
    s.add_argument('--print_every', type=int, default=100)
    s.add_argument('--device', default='cpu', help='CPU is the point; leave it')
    s.add_argument('--threads', type=int, default=None)
    s.set_defaults(func=cmd_stream)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

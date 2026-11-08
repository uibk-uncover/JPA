"""Tab. III: whether JPEG encoders pad unaligned images by repeating the edge.

For each encoder, setting, and image, the unaligned image and its padded version
are compressed, and their DCT coefficients compared.

Writes the repo columns of results/consistency.csv, next to the paper columns.

Usage:
    python3 run.py
    python3 run.py --color --qf 95 --pad 4
"""

import argparse
import cv2  #
import itertools
import jpeglib  #
import numpy as np  #
import pandas as pd  #
from pathlib import Path
from PIL import Image  #
import PIL  #
import sys
from tempfile import TemporaryDirectory

from compare_jpeg import compare_component, read_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'data'))
from download import DATA_DIR, get_precover


def pad(x: np.ndarray, block: int = 8) -> np.ndarray:
    """pads the bottom and right of an image to a multiple of the block size, by repeating the edge"""
    pad_width = [(0, -x.shape[0] % block), (0, -x.shape[1] % block)] + [(0, 0)] * (x.ndim - 2)
    return np.pad(x, pad_width, mode='edge')


def write_jpeglib(version):
    def write(x, path, qf):
        with jpeglib.version(version):
            jpeglib.from_spatial(x if x.ndim == 3 else x[..., None]).write_spatial(path, qt=qf)
    return write


def write_pillow(x, path, qf):
    Image.fromarray(x).save(path, quality=qf)


def write_opencv(x, path, qf):
    cv2.imwrite(str(path), x[..., ::-1] if x.ndim == 3 else x, [cv2.IMWRITE_JPEG_QUALITY, qf])


VERSIONS = sorted(jpeglib.version.versions())
ENCODERS = {
    'libjpeg': {v: write_jpeglib(v) for v in VERSIONS if v[0].isdigit()},
    'libjpeg-turbo': {v: write_jpeglib(v) for v in VERSIONS if v.startswith('turbo')},
    'MozJPEG': {v: write_jpeglib(v) for v in VERSIONS if v.startswith('mozjpeg')},
    'pillow': {PIL.__version__: write_pillow},
    'OpenCV': {cv2.__version__: write_opencv},
}


def is_repeat_padding(write, x, qf, tmp) -> bool:
    """compresses the unaligned and padded image, and compares the DCT coefficients"""
    path0, path1 = Path(tmp) / 'unaligned.jpeg', Path(tmp) / 'padded.jpeg'
    write(x, path0, qf)
    write(pad(x, block=16 if x.ndim == 3 else 8), path1, qf)
    (_, c0), (_, c1) = read_components(str(path0)), read_components(str(path1))
    return all((compare_component(c0[k], c1[k]) == 0).all() for k in c0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--color', action='store_true', help='use color images, grayscale by default')
    parser.add_argument('--qf', nargs='+', type=int, default=[75, 98], help='JPEG quality factors')
    parser.add_argument('--pad', nargs='+', type=int, default=list(range(1, 8)), help='square pad sizes')
    parser.add_argument('--num_images', type=int, default=2, help='number of images')
    parser.add_argument('--images', default=ROOT / 'data' / 'images.csv', type=Path, help='CSV with images (name, sha1)')
    parser.add_argument('--data', default=DATA_DIR, type=Path, help='cache for the precovers')
    args = parser.parse_args()

    images = pd.read_csv(args.images, dtype=str).head(args.num_images)
    precovers = [
        get_precover(row['name'], args.data, sha1=None if args.color else row['sha1'], color=args.color)
        for _, row in images.iterrows()
    ]

    repo = {}
    with TemporaryDirectory() as tmp:
        for encoder, versions in ENCODERS.items():
            consistent = []
            for version, write in versions.items():
                failed = [
                    (qf, pad_size)
                    for x0, qf, pad_size in itertools.product(precovers, args.qf, args.pad)
                    if not is_repeat_padding(write, x0[:x0.shape[0]-pad_size, :x0.shape[1]-pad_size], qf, tmp)
                ]
                print(f'{encoder} {version}: ' + ('repeat padding' if not failed else f'differs for (qf, pad) {sorted(set(failed))[:5]}'))
                consistent.append(not failed)
            repo[encoder] = ('-'.join([min(versions), max(versions)]) if len(versions) > 1 else next(iter(versions)), all(consistent))

    results = pd.read_csv(ROOT / 'results' / 'consistency.csv').set_index('encoder')
    results['version_repo'] = pd.Series({k: v for k, (v, _) in repo.items()})
    results['repeat_padding_repo'] = pd.Series({k: c for k, (_, c) in repo.items()})
    results.reset_index().to_csv(ROOT / 'results' / 'consistency.csv', index=False)

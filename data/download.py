"""Downloads the ALASKA precovers, and checks their hashes.

Usage:
    python3 download.py images.csv
"""

import argparse
import hashlib
import numpy as np  #
import pandas as pd  #
from pathlib import Path
from PIL import Image  #
from tqdm import tqdm  #
from urllib.request import urlopen

ALASKA_URL = 'https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_GrayScale/{}.tif'
ALASKA_COLOR_URL = 'https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_COLOR/{}.tif'
DATA_DIR = Path(__file__).resolve().parent / 'alaska'


def get_precover(name: str, data_dir: Path = DATA_DIR, sha1: str = None, color: bool = False) -> np.ndarray:
    """downloads the ALASKA precover, unless cached, and checks its hash, if given"""
    if color:
        data_dir = data_dir.parent / f'{data_dir.name}_color'
    path = data_dir / f'{name}.tif'
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen((ALASKA_COLOR_URL if color else ALASKA_URL).format(name), timeout=60).read())
    assert sha1 is None or hashlib.sha1(path.read_bytes()).hexdigest() == sha1, f'hash mismatch for {path}'
    return np.array(Image.open(path).convert('RGB' if color else 'L'))  # some precovers are 1-bit, e.g., 38018


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download of the ALASKA precovers.")
    parser.add_argument("images", type=Path, default='images.csv', help='CSV with images (stem or name, optionally sha1)')
    parser.add_argument("--data", default=DATA_DIR, type=Path, help='cache for the precovers')
    args = parser.parse_args()

    images = pd.read_csv(args.images, dtype=str).rename(columns={'stem': 'name'})
    for _, row in tqdm(images.iterrows(), total=len(images), desc='Downloading'):
        get_precover(row['name'], args.data, sha1=row.get('sha1'))
    print(f'{len(images)} images in {args.data}')

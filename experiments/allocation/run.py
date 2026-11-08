"""Average change rates in padded and fully supported blocks (Fig. 5).

Usage:
    python3 run.py --method UERD --qf 98 --padv 4 --padh 4 --alpha .4
"""

import os
for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(var, '1')  # one thread per process, images run in parallel

import argparse
from functools import partial
import jpeglib  #
from multiprocessing import Pool
import numpy as np  #
import pandas as pd  #
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / 'data'))
from download import DATA_DIR, get_precover
from _stego import decompress, probability


def process_image(row: dict, args: argparse.Namespace) -> tuple:
    """compresses, and returns the change rates in padded and fully supported blocks"""
    # compress with libjpeg 6b
    x0 = get_precover(row['name'], args.data, sha1=row['sha1'])
    x0 = x0[:x0.shape[0]-args.padv, :x0.shape[1]-args.padh]
    with NamedTemporaryFile(suffix='.jpeg') as tmp:
        jpeglib.from_spatial(x0[..., None]).write_spatial(tmp.name, qt=args.qf)
        jpeg = jpeglib.read_dct(tmp.name)
    y0, qt = jpeg.Y, jpeg.qt[jpeg.quant_tbl_no[0]]

    # change rates in the last block-row and block-column vs. the rest
    p_p1, p_m1 = probability(args.method, y0=y0, qt=qt, alpha=args.alpha, x0=decompress(y0, qt))
    beta = p_p1 + p_m1
    is_padded = np.zeros(beta.shape[:2], dtype='bool')
    is_padded[-1, :] = args.padv > 0
    is_padded[:, -1] |= args.padh > 0
    return beta[is_padded].mean(), beta[~is_padded].mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Allocation of changes into padded blocks.")
    parser.add_argument("--method", required=True, choices=['nsF5', 'EBS', 'JUNIWARD', 'UERD'], help='steganographic method')
    parser.add_argument("--alpha", default=.4, type=float, help='embedding rate [bits per coefficient]')
    parser.add_argument("--qf", default=98, type=int, help='JPEG quality factor')
    parser.add_argument("--padv", default=4, type=int, help='vertical pad size')
    parser.add_argument("--padh", default=4, type=int, help='horizontal pad size')
    parser.add_argument("--images", default=Path(__file__).resolve().parents[2] / 'data' / 'images.csv', type=Path, help='CSV with images (name, sha1)')
    parser.add_argument("--data", default=DATA_DIR, type=Path, help='cache for the precovers')
    parser.add_argument("--output", default=Path(__file__).resolve().parents[2] / 'results' / 'allocation.csv', type=Path, help='CSV with results')
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help='number of parallel processes')
    args = parser.parse_args()

    # process images in parallel
    rows = pd.read_csv(args.images, dtype=str).to_dict('records')
    with Pool(args.workers) as pool:
        beta_pad, beta_full = zip(*tqdm(pool.imap(partial(process_image, args=args), rows), total=len(rows)))

    # write the repo columns of this setting, next to the paper columns
    key = {'stego_method': args.method, 'alpha': args.alpha, 'qf': args.qf, 'padv': args.padv, 'padh': args.padh}
    repo = {'beta_pad_C_repo': np.mean(beta_pad), 'beta_full_C_repo': np.mean(beta_full)}
    results = pd.read_csv(args.output)
    is_setting = (results[list(key)] == pd.Series(key)).all(axis=1)
    assert is_setting.any(), f'setting {key} not in {args.output}'
    results.loc[is_setting, list(repo)] = list(repo.values())
    results.to_csv(args.output, index=False)
    print(results[is_setting].to_string(index=False))

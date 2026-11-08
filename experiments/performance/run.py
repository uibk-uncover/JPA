"""Detection error of JPA, RJCA, and SCA-RJCA (Fig. 6-7, Tab. IV).

Usage:
    python3 run.py --method UERD --qf 98 --padv 4 --padh 4 --alpha .4
"""

import os
for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(var, '1')  # one thread per process, images run in parallel

import argparse
import conseal as cl  #
from functools import partial
import jpeglib  #
from multiprocessing import Pool
import numpy as np  #
import pandas as pd  #
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from tqdm import tqdm

from attack import calculate_jpa

sys.path.append(str(Path(__file__).resolve().parents[2] / 'data'))
from download import DATA_DIR, get_precover
from _stego import decompress, embed, probability


def calculate_rjca(y: np.ndarray, qt: np.ndarray, beta: np.ndarray = None) -> float:
    """variance of the rounding error after decompression, over all blocks

    :param y: quantized DCT coefficients
    :param qt: quantization table
    :param beta: change probabilities, for the selection-channel-aware variant (Cogranne'20)
    """
    x = cl.tools.dct.block_idct2(y * qt[None, None])
    e = x - np.round(x)
    # original RJCA
    if beta is None:
        return np.var(e)
    # SCA-RJCA, weighted by the change rate of each block
    return np.sum(beta.sum(axis=(2, 3)) * (e**2).sum(axis=(2, 3))) / (np.sum(beta) + 1e-12)


def compute_pe(scores0: np.ndarray, scores1: np.ndarray, num_steps: int = 100) -> float:
    """minimal probability of error over quantile thresholds"""
    scores = np.r_[scores0, scores1]
    taus = np.r_[np.quantile(scores, np.linspace(0., 1., num_steps + 1)), np.max(scores) + .1]
    p_fp = (scores0[:, None] >= taus[None, :]).mean(axis=0)
    p_tp = (scores1[:, None] >= taus[None, :]).mean(axis=0)
    return np.min(.5 * (p_fp + 1 - p_tp))


def process_image(row: dict, args: argparse.Namespace) -> dict:
    """compresses, embeds, and returns the cover and stego scores of all attacks"""
    # compress with libjpeg 6b
    x0 = get_precover(row['name'], args.data, sha1=row['sha1'])
    x0 = x0[:x0.shape[0]-args.padv, :x0.shape[1]-args.padh]
    with NamedTemporaryFile(suffix='.jpeg') as tmp:
        jpeglib.from_spatial(x0[..., None]).write_spatial(tmp.name, qt=args.qf)
        jpeg = jpeglib.read_dct(tmp.name)
    y0, qt, shape = jpeg.Y, jpeg.qt[jpeg.quant_tbl_no[0]], (jpeg.height, jpeg.width)

    # embed, with an image-specific seed
    # the paper used seed=12345 for all images, which correlates the changes across images;
    # P_E differs within sampling noise (e.g., UERD at QF 98: .138 in the paper, .130 here)
    seed = cl.tools.password_to_seed(row['name'])
    y1, ps0 = embed(args.method, y0=y0, qt=qt, alpha=args.alpha, seed=seed, x0=decompress(y0, qt))

    # selection channel, estimated by the attacker from the image at hand
    ps1 = probability(args.method, y0=y1, qt=qt, alpha=args.alpha, x0=decompress(y1, qt))

    # detect
    images = [(y0.astype('float64'), sum(ps0)), (y1.astype('float64'), sum(ps1))]
    qt = qt.astype('float64')
    return {
        'jpa': [calculate_jpa(y, qt=qt, shape=shape, beta=beta) for y, beta in images],
        'rjca': [calculate_rjca(y, qt=qt) for y, _ in images],
        'rjca_sca': [calculate_rjca(y, qt=qt, beta=beta) for y, beta in images],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performance of JPA, RJCA, and SCA-RJCA.")
    parser.add_argument("--method", required=True, choices=['nsF5', 'EBS', 'JUNIWARD', 'UERD'], help='steganographic method')
    parser.add_argument("--alpha", default=.4, type=float, help='embedding rate [bits per coefficient]')
    parser.add_argument("--qf", default=98, type=int, help='JPEG quality factor')
    parser.add_argument("--padv", default=4, type=int, help='vertical pad size')
    parser.add_argument("--padh", default=4, type=int, help='horizontal pad size')
    parser.add_argument("--images", default=Path(__file__).resolve().parents[2] / 'data' / 'images.csv', type=Path, help='CSV with images (name, sha1)')
    parser.add_argument("--data", default=DATA_DIR, type=Path, help='cache for the precovers')
    parser.add_argument("--output", default=Path(__file__).resolve().parents[2] / 'results' / 'performance.csv', type=Path, help='CSV with results')
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help='number of parallel processes')
    args = parser.parse_args()

    # process images in parallel
    rows = pd.read_csv(args.images, dtype=str).to_dict('records')
    with Pool(args.workers) as pool:
        scores = list(tqdm(pool.imap(partial(process_image, args=args), rows), total=len(rows)))

    # write the repo columns of this setting, next to the paper columns
    key = {'stego_method': args.method, 'alpha': args.alpha, 'qf': args.qf, 'padv': args.padv, 'padh': args.padh}
    repo = {f'pe_{attack}_repo': compute_pe(*np.array([s[attack] for s in scores]).T) for attack in scores[0]}
    results = pd.read_csv(args.output)
    is_setting = (results[list(key)] == pd.Series(key)).all(axis=1)
    assert is_setting.any(), f'setting {key} not in {args.output}'
    results.loc[is_setting, list(repo)] = list(repo.values())
    results.to_csv(args.output, index=False)
    print(results[is_setting].to_string(index=False))

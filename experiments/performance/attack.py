"""demonstration of the JPA attack"""

import argparse
import conseal as cl  #
import jpeglib  #
import numpy as np  #
from pathlib import Path
from PIL import Image  #
import sys
from tempfile import NamedTemporaryFile
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[2] / 'data'))
from _stego import decompress, embed, probability


def get_weights(qt: np.ndarray, p_ref: int, axis: int = 0) -> np.ndarray:
    """calculates the JPA weights as DCT-base differences

    this is needed both for residuals (Eq. 5) and for variances (Eq. 6-7)

    :param qt: quantization table
    :param p_ref: edge index
    :param axis: orientation
    """
    #
    [col, row] = np.meshgrid(range(8), range(8))  # 8 8
    dct_1d = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))  # 8 8
    dct_1d[0, :] = dct_1d[0, :] / np.sqrt(2)  # 8 8
    #
    all_bases = np.einsum('mr, nc -> rcmn', dct_1d, dct_1d)  # vs hs vf hf
    """r/c are spatial m/n are frequency"""

    # all vs. all
    indices = np.arange(p_ref, 8)
    idx1, idx2 = np.triu_indices(len(indices), k=1)
    if axis == 0:
        basis1 = all_bases[indices[idx1], :]
        basis2 = all_bases[indices[idx2], :]
    else:
        basis1 = all_bases[:, indices[idx1]]
        basis2 = all_bases[:, indices[idx2]]
    weights = basis2 - basis1
    return weights**2 * qt.astype('float64')**2  # vs hs vf hf
    """the order is rcmn"""


def idct_2d(y: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """2D inverse DCT transform

    :param y: quantized DCT coefficients
    :param qt: quantization table
    """
    # inverse DCT
    [col, row] = np.meshgrid(range(8), range(8))  # 8 8
    dct_1d = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))  # 8 8
    dct_1d[0, :] = dct_1d[0, :] / np.sqrt(2)  # 8 8
    dct_1d_left = (dct_1d.T)[None, None, :, :]  # 1 1 vf vs
    dct_1d_right = dct_1d[None, None, :, :]  # 1 1 hs hf
    x = dct_1d_left @ (y * qt[None, None]) @ dct_1d_right  # Hb Wb 8 8
    #
    return x


def get_residuals(
    y: np.ndarray,
    qt: np.ndarray,
    shape: Tuple[int],
) -> Tuple[np.ndarray]:
    """calculates residuals for JPA attack (Eq. 5)

    :param y: quantized DCT coefficients
    :param qt: quantization table
    :param shape: image size with support
    """
    H, W = [(s - 1) % 8 for s in shape]

    # inverse DCT
    x = idct_2d(y, qt)

    # vertical
    v_indices = np.arange(H, 8)
    x_pad_v = x[-1:, :, v_indices, :]
    if x_pad_v.shape[1] > 1 and H > 0:
        x_pad_v = x_pad_v[:, :-1]
    idx1_v, idx2_v = np.triu_indices(len(v_indices), k=1)
    r_v = x_pad_v[:, :, idx2_v, :] - x_pad_v[:, :, idx1_v, :]

    # horizontal
    h_indices = np.arange(W, 8)
    x_pad_h = x[:, -1:, :, h_indices]
    if x_pad_h.shape[0] > 1 and W > 0:
        x_pad_h = x_pad_h[:-1]
    idx1_h, idx2_h = np.triu_indices(len(h_indices), k=1)
    r_h = x_pad_h[:, :, :, idx2_h] - x_pad_h[:, :, :, idx1_h]
    #
    return r_v, r_h


def get_variances(
    y: np.ndarray,
    qt: np.ndarray,
    shape: Tuple[int],
    beta: np.ndarray,
) -> Tuple[Tuple[np.ndarray]]:
    """calculates cover and stego variances (Eq. 6-7)

    :param y: quantized DCT coefficients
    :param qt: quantization tables
    :param shape: image size with support
    :param beta: binary change probabilities
    """
    H, W = [(s - 1) % 8 for s in shape]
    # vertical
    beta_v = beta[-1:]  # 1 Hb vf hf
    if beta_v.shape[1] > 1 and H > 0:
        beta_v = beta_v[:, :y.shape[1]-1]  # 1 Hb-1 vf hf
    weights_v = get_weights(qt, H, axis=0)  # vs hs vf hf
    sigma2_0_v = (1/12) * np.sum(weights_v, axis=(2, 3))[None, None]  # 1 1 vs hs
    sigma2_0_v = np.repeat(sigma2_0_v, beta_v.shape[1], axis=1)  # 1 Hb-1 vs hs
    sigma2_1_v = sigma2_0_v + np.einsum('hwmn, rcmn -> hwrc', beta_v, weights_v)  # 1 Hb-1 vs hs
    # horizontal
    beta_h = beta[:, -1:]  # Vb 1 vf hf
    if beta_h.shape[0] > 1 and W > 0:
        beta_h = beta_h[:y.shape[0]-1]  # Vb-1 1 vf hf
    weights_h = get_weights(qt, W, axis=1)  # vs hs vf hf
    sigma2_0_h = (1/12) * np.sum(weights_h, axis=(2, 3))[None, None]  # 1 1 vs hs
    sigma2_0_h = np.repeat(sigma2_0_h, beta_v.shape[0], axis=0)  # Vb-1 1 vs hs
    sigma2_1_h = sigma2_0_h + np.einsum('hwmn, rcmn -> hwrc', beta_h, weights_h)  # Vb-1 1 vs hs
    #
    return (sigma2_0_v, sigma2_0_h), (sigma2_1_v, sigma2_1_h)


def calculate_statistic(r: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> float:
    """calculates log-likelihood ratio (Eq. 8)

    :param r: residuals
    :param s0: cover variances
    :param s1: stego variances
    """
    return 0.5 * np.sum(np.log(s0 / s1) + r**2 * (1 / s0 - 1 / s1))


def calculate_jpa(y: np.ndarray, qt: np.ndarray, shape: Tuple[int], beta: np.ndarray) -> float:
    """calculates the attack score over horizontal and vertical blocks

    :param y: quantized DCT coefficients
    :param qt: quantization tables
    :param shape: image size with support
    :param beta: binary change probabilities
    """
    # Eq. 5
    r_v, r_h = get_residuals(y, qt=qt, shape=shape)
    # Eq 6-7
    (sigma2_0_v, sigma2_0_h), (sigma2_1_v, sigma2_1_h) = get_variances(
        y,
        beta=beta,
        qt=qt,
        shape=shape)
    # Eq. 8
    lrt = (
        # vertical
        +calculate_statistic(r_v, sigma2_0_v, sigma2_1_v)
        # horizontal
        +calculate_statistic(r_h, sigma2_0_h, sigma2_1_h)
        # the (corner) diagonal block can be used too, but we ignore it in this code
    )
    return lrt
    #


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description="Demonstration of the JPEG padding attack (JPA).")
    parser.add_argument("--precover", required=True, type=Path, help='path to precover')
    parser.add_argument("--method", default='nsF5', choices=['nsF5', 'EBS', 'JUNIWARD', 'UERD'], help='steganographic method')
    parser.add_argument("--alpha", default=.4, type=float, help='embedding rate [bits per coefficient]')
    parser.add_argument("--qf", default=98, type=float, help='JPEG quality factor')
    parser.add_argument("--pad", default=4, type=int, help='(square) pad size')
    parser.add_argument("--seed", default=None, type=int, help='stego seed, derived from the file name by default')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = cl.tools.password_to_seed(args.precover.stem)

    # prepare pre-cover
    x0 = np.array(Image.open(args.precover).convert('L'))
    x0_crop = x0[:-args.pad, :-args.pad]  # square pad

    # compress
    with NamedTemporaryFile(suffix='.jpeg') as tmp:
        jpeglib.from_spatial(x0_crop[..., None]).write_spatial(tmp.name, qt=args.qf)
        jpeg = jpeglib.read_dct(tmp.name)
    y0, qt = jpeg.Y, jpeg.qt[jpeg.quant_tbl_no[0]]  # DCT and QT
    shape = jpeg.height, jpeg.width

    # embed
    y1, (p_p1, p_m1) = embed(args.method, y0=y0, qt=qt, alpha=args.alpha, seed=args.seed, x0=decompress(y0, qt))
    print('Embedded %s at %.4f bpC.' % (args.method, cl.tools.entropy(p_p1, p_m1) / y0.size,))

    # selection channel, estimated by the attacker from the image at hand
    beta0 = p_p1 + p_m1
    beta1 = sum(probability(args.method, y0=y1, qt=qt, alpha=args.alpha, x0=decompress(y1, qt)))

    # detect
    jpa0 = calculate_jpa(y=y0, qt=qt, shape=shape, beta=beta0)
    print('JPEG padding attack:')
    print('* cover score %.04f' % (jpa0,))
    jpa1 = calculate_jpa(y=y1, qt=qt, shape=shape, beta=beta1)
    print('* stego score %.04f' % (jpa1,))

    # ... run this over 500 images and calculate the PE ...

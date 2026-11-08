"""Change probabilities and simulated embedding."""

import conseal as cl  #
import numpy as np  #
from typing import Tuple


def decompress(y: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """decompresses the analyzed image over the full block grid, including the padding, unrounded, for J-UNIWARD"""
    x = cl.tools.dct.block_idct2(y * qt[None, None])  # Hb Wb 8 8
    return x.transpose(0, 2, 1, 3).reshape(x.shape[0] * 8, x.shape[1] * 8)


def probability(
    method: str,
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    x0: np.ndarray = None,
) -> Tuple[np.ndarray]:
    """calculates change probabilities of +1 and -1

    :param method: steganographic method, one of nsF5, EBS, JUNIWARD, UERD
    :param y0: quantized cover DCT coefficients
    :param qt: quantization table
    :param alpha: embedding rate [bits per coefficient]
    :param x0: decompressed image, required for J-UNIWARD, see decompress()
    """
    # no embedding into flat images, e.g., 38018, as in the B0 training data
    if cl.tools.nzAC(y0) == 0:
        return np.zeros(y0.shape), np.zeros(y0.shape)
    if method == 'nsF5':
        # conseal expects bits per nzAC
        alpha_nzAC = alpha * y0.size / cl.tools.nzAC(y0)
        (p_p1, p_m1), _ = cl.nsF5.probability(y0, alpha_nzAC)
        return p_p1, p_m1
    elif method == 'EBS':
        rhos = cl.ebs.compute_cost_adjusted(y0, qt)
    elif method == 'JUNIWARD':
        assert x0 is not None, 'J-UNIWARD needs the decompressed image'
        # original implementation, as in the paper (default of conseal before 2026.11)
        rhos = cl.juniward.compute_cost_adjusted(x0, y0, qt, implementation=cl.juniward.Implementation.JUNIWARD_ORIGINAL)
    elif method == 'UERD':
        rhos = cl.uerd.compute_cost_adjusted(y0, qt)
    else:
        raise NotImplementedError(f'unknown method {method}')
    # payload-limited sender, λ by binary search
    (p_p1, p_m1), _ = cl.simulator.probability(
        rhos,
        alpha=alpha,
        n=y0.size,  # bits per coefficient
        solver=cl.simulator.SOLVER_BSEARCH_DDE,
    )
    return p_p1, p_m1


def embed(
    method: str,
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    seed: int,
    x0: np.ndarray = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """simulates embedding and returns stego DCT coefficients and change probabilities

    the change probabilities are returned to be reused by the attack,
    since computing the costs is the slowest part

    :param method: steganographic method, one of nsF5, EBS, JUNIWARD, UERD
    :param y0: quantized cover DCT coefficients
    :param qt: quantization table
    :param alpha: embedding rate [bits per coefficient]
    :param seed: stego key
    :param x0: decompressed image, required for J-UNIWARD, see decompress()
    """
    ps = probability(method, y0=y0, qt=qt, alpha=alpha, x0=x0)
    delta = cl.simulator.sample(ps, seed=seed)
    return y0 + delta, ps

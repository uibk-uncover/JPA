"""Compares the DCT coefficients of two JPEGs; exits with 0 if identical.

Usage:
    python3 compare_jpeg.py ../../data/consistency/color_unaligned.jpg ../../data/consistency/color_padded.jpg
    python3 compare_jpeg.py ../../data/consistency/color_unaligned.jpg ../../data/consistency/color_padded.jpg --show
"""

import argparse
import sys

import jpeglib
import numpy as np

COMPONENTS = ['Y', 'Cb', 'Cr']


def read_components(path: str) -> tuple:
    """Reads the quantized DCT coefficients of a JPEG.

    :param path: path to the JPEG file
    :return: (jpeglib DCT object, dict of component name -> ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8])
    """
    jpeg = jpeglib.read_dct(path)
    components = {'Y': jpeg.Y}
    if jpeg.has_chrominance:
        components['Cb'] = jpeg.Cb
        components['Cr'] = jpeg.Cr
    return jpeg, components


def compare_component(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Coefficient-wise differences over the common block grid.

    :return: c1 - c0 of shape [common_vertical_blocks, common_horizontal_blocks, 8, 8]
    """
    h = min(c0.shape[0], c1.shape[0])
    w = min(c0.shape[1], c1.shape[1])
    return c1[:h, :w].astype(int) - c0[:h, :w].astype(int)


def report_component(name: str, diff: np.ndarray) -> None:
    """Prints a summary of differences of a single component."""
    differing = (diff != 0).any(axis=(2, 3))
    h, w = differing.shape
    last_row = differing[-1, :].sum()
    last_col = differing[:, -1].sum()
    corner = differing[-1, -1]
    interior = differing[:-1, :-1].sum()
    print(f'{name}: {differing.sum()} / {h * w} blocks differ'
          f' (last block-row {last_row}, last block-column {last_col}, of which corner {int(corner)}; interior {interior})')
    if differing.any():
        absdiff = np.abs(diff)
        print(f'  coefficients differing: {(diff != 0).sum()},'
              f' max |diff|: {absdiff.max()}, mean |diff| in differing blocks: {absdiff[differing].mean():.3f}')


def plot(path0: str, diffs: dict) -> None:
    """Visualizes the differences.

    Left: the first image, with differing blocks marked in red.
    Per component: map of differing blocks (sum of |diff| per block),
    and mean |diff| per DCT frequency over the differing blocks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    x0 = jpeglib.read_spatial(path0).spatial
    if x0.shape[-1] == 1:
        x0 = np.repeat(x0, 3, axis=-1)
    x0 = x0.copy()
    mask = np.zeros(x0.shape[:2], dtype=bool)
    for diff in diffs.values():
        differing = (diff != 0).any(axis=(2, 3))
        # subsampled chroma blocks cover more than 8x8 pixels
        bh = 8 * int(np.ceil(x0.shape[0] / 8 / differing.shape[0]))
        bw = 8 * int(np.ceil(x0.shape[1] / 8 / differing.shape[1]))
        upsampled = np.kron(differing, np.ones((bh, bw), dtype=bool))[:x0.shape[0], :x0.shape[1]]
        mask[:upsampled.shape[0], :upsampled.shape[1]] |= upsampled
    x0[mask] = [255, 0, 0]

    n = len(diffs)
    fig = plt.figure(figsize=(10, max(4, 2.4 * n)), constrained_layout=True)
    gs = GridSpec(n, 3, figure=fig, width_ratios=[1.3, 1, .8])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(x0)
    ax.set_title(f'{path0}\ndiffering blocks in red', fontsize=9)
    ax.axis('off')
    for k, (name, diff) in enumerate(diffs.items()):
        absdiff = np.abs(diff)
        block_diff = absdiff.sum(axis=(2, 3))
        differing = block_diff > 0

        ax = fig.add_subplot(gs[k, 1])
        ax.imshow(block_diff, cmap='Reds', vmin=0, vmax=max(block_diff.max(), 1), interpolation='nearest')
        # single blocks are invisible in a large map, so also scatter them
        ii, jj = np.nonzero(differing)
        ax.scatter(jj, ii, s=6, c='red', marker='s')
        ax.set_xlim(-.5, diff.shape[1] - .5)
        ax.set_ylim(diff.shape[0] - .5, -.5)
        ax.set_title(f'{name}: sum |diff| per block ({differing.sum()} differing)', fontsize=9)
        ax.set_xlabel('block column', fontsize=8)
        ax.set_ylabel('block row', fontsize=8)

        ax = fig.add_subplot(gs[k, 2])
        freq_diff = absdiff[differing].mean(axis=0) if differing.any() else np.zeros((8, 8))
        im = ax.imshow(freq_diff, cmap='Reds', vmin=0, vmax=freq_diff.max() or 1, interpolation='nearest')
        for (u, v), val in np.ndenumerate(freq_diff):
            if val > 0:
                ax.text(v, u, f'{val:.1f}', ha='center', va='center', fontsize=6)
        ax.set_title(f'{name}: mean |diff| per frequency', fontsize=9)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        fig.colorbar(im, ax=ax, shrink=.8)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('jpeg0', help='first JPEG, e.g. compressed by the encoder with its own padding')
    parser.add_argument('jpeg1', help='second JPEG, e.g. compressed after padding by repeating the edge')
    parser.add_argument('--show', action='store_true', help='show the differences')
    args = parser.parse_args()

    jpeg0, components0 = read_components(args.jpeg0)
    jpeg1, components1 = read_components(args.jpeg1)
    print(f'{args.jpeg0}: {jpeg0.height}x{jpeg0.width}, components {list(components0)}, blocks {jpeg0.Y.shape[:2]}')
    print(f'{args.jpeg1}: {jpeg1.height}x{jpeg1.width}, components {list(components1)}, blocks {jpeg1.Y.shape[:2]}')
    if jpeg0.Y.shape[:2] != jpeg1.Y.shape[:2]:
        print('warning: different block grids, comparing the common part')
    if not np.array_equal(jpeg0.qt, jpeg1.qt):
        print('warning: different quantization tables, coefficients are not directly comparable')
    if list(components0) != list(components1):
        print('warning: different components, comparing the common ones')

    diffs = {}
    for name in COMPONENTS:
        if name in components0 and name in components1:
            diffs[name] = compare_component(components0[name], components1[name])
            report_component(name, diffs[name])

    identical = all((diff == 0).all() for diff in diffs.values())
    print('identical DCT coefficients' if identical else 'DCT coefficients differ')

    if args.show:
        plot(args.jpeg0, diffs)

    sys.exit(0 if identical else 1)


if __name__ == '__main__':
    main()

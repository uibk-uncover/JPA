"""Fig. 2-3 and Sec. II-B, VI: image formats, JPEG padding sizes, and JPEG qualities in the web crawl.

Writes the repo columns of results/prevalence_formats.csv and results/prevalence_padding.csv,
next to the paper columns, and prints the values stated in the text.

Usage:
    python3 run.py
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=ROOT / 'data' / 'prevalence' / 'images.csv.gz', type=Path, help='CSV with the crawled images')
    parser.add_argument('--results', default=ROOT / 'results', type=Path, help='directory with the results')
    args = parser.parse_args()

    df = pd.read_csv(args.images, low_memory=False)
    jpeg = df[df['format'] == 'JPEG']
    is_tiny = ~((jpeg['width'] >= 8) & (jpeg['height'] >= 8))
    width, height = jpeg['width'][~is_tiny].astype(int), jpeg['height'][~is_tiny].astype(int)
    is_aligned = (width % 8 == 0) & (height % 8 == 0)

    # Fig. 2
    counts = df['format'].value_counts()
    counts['JPEG (tiny)'] = is_tiny.sum()
    counts['JPEG (aligned)'] = is_aligned.sum()
    counts['JPEG (unaligned)'] = (~is_aligned).sum()
    formats = pd.read_csv(args.results / 'prevalence_formats.csv')
    formats['count_repo'] = formats['format'].map(counts)
    formats.to_csv(args.results / 'prevalence_formats.csv', index=False)

    # Fig. 3
    padding = (
        pd.crosstab(height % 8, width % 8, normalize='all')
        .rename_axis(index='padv', columns='padh')
        .stack()
        .mul(100)
        .round(2)
    )
    results = pd.read_csv(args.results / 'prevalence_padding.csv').set_index(['padv', 'padh'])
    results['prevalence_repo'] = padding
    results.reset_index().to_csv(args.results / 'prevalence_padding.csv', index=False)

    # text, Sec. II-B
    print(f'unique images: {len(df)} (paper: 1.6 million)')
    print(f'tiny JPEGs: {is_tiny.mean():.1%} (paper: 8%)')
    print(f'unaligned JPEGs: {(~is_aligned).mean():.1%} (paper: 80%)')

    # text, Sec. VI, from the decoded JPEGs
    is_high = jpeg['qt'][~is_tiny].isin(['standard', 'libjpeg']) & (jpeg['qf'][~is_tiny] >= 98)
    print(f'unaligned JPEGs with QF >= 98: {(is_high & ~is_aligned).sum() / (~is_tiny).sum():.1%} (paper: 2%)')

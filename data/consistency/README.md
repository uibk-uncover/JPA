# Lightroom example

A test pair for the consistency experiment, and its export by Adobe Lightroom, which does not use repeat padding (Tab. III).

| File | Content |
|---|---|
| `color_unaligned.png` | photo of 508x508 pixels (padding 4x4), downsampled from the photographer's own image |
| `color_padded.png` | the same, padded to 512x512 by repeating the edge |
| `gray_unaligned.png`, `gray_padded.png` | grayscale version of the pair |
| `*.jpg` | the PNGs exported by Adobe Lightroom, with the same settings |

The DCT coefficients of the Lightroom exports differ by +-1 in most of the padded blocks (last block-row and block-column), and nowhere else.
Compressed with libjpeg, the coefficients of both images are identical.

```bash
python3 experiments/consistency/compare_jpeg.py data/consistency/color_unaligned.jpg data/consistency/color_padded.jpg --show
python3 experiments/consistency/compare_jpeg.py data/consistency/gray_unaligned.jpg data/consistency/gray_padded.jpg --show
```

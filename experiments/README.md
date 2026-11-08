# Differences to the paper

Each results CSV holds the values from the paper (`_paper`) next to the values recomputed by this repository (`_repo`).
This file explains where and why they differ.


## Prevalence

The crawl list in `data/prevalence/images.csv.gz` was exported after the camera-ready version, from a later state of the crawl, to collect additional details (color, quantization table, and quality).
The aggregates match the paper closely, but not exactly.

| Statistic | Paper | Repo |
|---|---|---|
| Formats (Fig. 2), in 10^5: JPEG, PNG, SVG, WebP, GIF | 7.4, 5.3, 2.5, .61, .51 | 7.19, 5.07, 2.28, .58, .44 |
| JPEGs smaller than one block or without size | 8% | 7.9% |
| Unaligned JPEGs (Fig. 3) | 80% | 79.6% |
| Padding sizes (Fig. 3), per cell | `prevalence_paper` in `results/prevalence_padding.csv` | `prevalence_repo`, within 0.13 %-pts |
| Unaligned JPEGs with standard QT and QF >= 98 (Sec. VI) | 2% | 2.1% |


## Consistency

The encoders in jpeglib, pillow, and OpenCV are retested automatically and all use repeat padding, as in Tab. III.
Pillow is tested in version 12.2.0 instead of 12.1.0.
ImageMagick, darktable, Gimp, and Lightroom are tested manually and have no repo values.


## Allocation

The change rates match the paper exactly for EBS, UERD, and nsF5, except for the payload sweep of nsF5 at QF 99 (alpha 0.05-0.3 and 0.5).
In the paper, these points were simulated with alpha in bits per nonzero AC coefficient instead of bits per coefficient, i.e., at a lower payload.

J-UNIWARD costs use the original implementation (`JUNIWARD_ORIGINAL`, the default of conseal before 2026.11) on the decompressed image, edge-padded to the block grid (`decompress` in `data/_stego.py`).


## Performance


After the camera-ready submission, we (with the help of Claude Opus 5.5) identified several bugs in the codebase.
Their impact on the conclusions is minor and, with the exception of nsF5 at QF 99, the issues work against JPA.
With the corrections, its detection improves and the B0 baseline gets weaker.


### JPA, RJCA, and SCA-RJCA

- **Cover change probabilities.**
  In the paper, JPA and SCA-RJCA always used the change probabilities derived from the cover, also for the stego.
  The attacker does not know these, and should always derive the change probabilities from the input image (cover or stego).
  This slightly improves SCA-RJCA for EBS (by 0.005), for other methods it remains within noise.
  For JPA, the P_E of most methods stays within noise, in both directions, except for nsF5, which JPA detects better, in particular at lower QFs.
  As a result, its improvement margin to SCA-RJCA is larger than reported in the paper for nsF5 at lower QFs.
- **Seeds.**
  The paper used the stego key 12345 for all images, which correlates the changes across images.
  Here, each image has its own key, derived from its name.
  This only happened for the statistical attacks, which do not have the means to exploit this regularity.
  It resulted in slight changes of P_E within sampling noise, in both directions.
- **nsF5 at QF 99.**
  For nsF5 at QF 99, the paper accidentally reports results from an earlier run, which used the embedding rate in bits per nonzero AC coefficient, i.e., a lower payload.
  Therefore, the paper's P_E of RJCA and SCA-RJCA are about .04 too high there, and at alpha 0.2 and 0.3, SCA-RJCA, not JPA, is the best attack.


### B0

- **Test images.**
  The paper evaluated the detectors on the 500 images in `data/images.csv`, of which 360 are in the training split and 71 in the validation split of `models/`.
  On these 500 images, this repository reproduces the paper's P_E within .003, see `results/performance_b0_500.csv`.
  `results/performance_b0.csv` holds P_E on the held-out test split (12001 images) instead, which is about .01 higher.
  The evaluation on the 500 images is run with

  ```bash
  python3 experiments/performance/evaluate_b0.py --method UERD --qf 98 \
      --images data/images.csv --output results/performance_b0_500.csv
  ```
- **nsF5.**
  The paper's nsF5 detectors were trained and tested on stegos with a higher payload than reported, due to a bug.
  The detectors in `models/` are retrained on correct stegos.
  Detection remains near-perfect nonetheless, with P_E .006 (QF 95) and .016 (QF 98) on the held-out test split, instead of 0 in the paper.

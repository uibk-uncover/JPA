# Steganalysis Exploiting JPEG Block Padding

This is the source repository for the paper "Steganalysis Exploiting JPEG Block Padding", IEEE WIFS 2026.

Set it up with

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```


## Content

The repository follows the experiments in Table II of the paper.

| Experiment | Paper | Code | Results |
|---|---|---|---|
| Prevalence | Sec. II-B, Fig. 2-3, Sec. VI | `experiments/prevalence/` | `results/prevalence_formats.csv`, `results/prevalence_padding.csv` |
| Consistency | Sec. V-B, Tab. III | `experiments/consistency/` | `results/consistency.csv` |
| Allocation | Sec. V-C, Fig. 5 | `experiments/allocation/` | `results/allocation.csv` |
| Performance | Sec. V-D, Fig. 6-7, Tab. IV | `experiments/performance/` | `results/performance.csv`, `results/performance_b0.csv`, `results/performance_b0_500.csv` |

Each results CSV holds the values from the paper (`_paper`) next to the values recomputed by this repository (`_repo`).
The differences are explained in `experiments/README.md`.


## Data

| Data | Used by | Location |
|---|---|---|
| 500 ALASKA v2 images (grayscale 512x512 TIFF), with SHA-1 | consistency, allocation, performance | listed in `data/images.csv` |
| Full ALASKA v2 (80005 images), split into training, validation, and test | B0 detectors | listed in `models/split_{tr,va,te}.csv` |
| Web crawl by Dornauer and Felderer (2023) | prevalence | `data/prevalence/images.csv.gz`, see `data/prevalence/README.md` |
| Test pair exported by Adobe Lightroom | consistency | `data/consistency/`, see `data/consistency/README.md` |
| Pre-trained B0 detectors | performance | [GitHub release](https://github.com/uibk-uncover/JPA/releases/latest), downloaded to `models/` on first use |

The ALASKA images are downloaded from the [ALASKA website](https://alaska.utt.fr/) to `data/alaska/` on first use (color images to `data/alaska_color/`), or in advance with

```bash
python3 data/download.py data/images.csv
```

Evaluating the B0 detectors on their test split downloads its 12001 images.


## Attack

The JPEG padding attack (JPA) itself for a single image is implemented in `experiments/performance/attack.py`.
You can try it out as follows.

```bash
python3 experiments/performance/attack.py \
    --precover "/home/username/cover.png" \
    --method nsF5 \
    --alpha .4 \
    --qf 98 \
    --pad 4
```


## Experiments

### Prevalence

The prevalence is measured on the web crawl by Dornauer and Felderer (2023).
The anonymized list of the crawled images is in `data/prevalence/images.csv.gz`.
It can be used to analyze the image formats (Fig. 2), padding sizes (Fig. 3), and JPEG qualities (Sec. VI) are computed from it.
The format counts and padding sizes are written to `results/prevalence_formats.csv` and `results/prevalence_padding.csv`, next to the values in the paper; the values from the text are printed.

```bash
python3 experiments/prevalence/run.py
```

### Consistency

For each encoder, we crop an image to unaligned dimensions, pad it with repeated edge pixels, compress both with the encoder, and compare the DCT coefficients.
The encoders in jpeglib (libjpeg, libjpeg-turbo, MozJPEG), pillow, and OpenCV are tested automatically, written to `results/consistency.csv` next to Tab. III.

```bash
python3 experiments/consistency/run.py  # grayscale, square pad sizes 1-7, QF 75 and 98
```

Other encoders are tested manually, with the image pair in `data/consistency/`.
It also contains a pair of images exported by Adobe Lightroom, which does not use repeat padding.

```bash
# compress data/consistency/color_unaligned.png -> unaligned.jpg and color_padded.png -> padded.jpg with the encoder under test
python3 experiments/consistency/compare_jpeg.py unaligned.jpg padded.jpg --show  # plots the differing blocks in red
```

### Allocation

Average change rates in padded and fully supported blocks, for one setting, written to `results/allocation.csv`.

```bash
python3 experiments/allocation/run.py --method UERD --qf 98 --padv 4 --padh 4 --alpha .4
```

### Performance

Detection error of JPA, RJCA, and SCA-RJCA, for one setting, written to `results/performance.csv`.

```bash
python3 experiments/performance/run.py --method UERD --qf 98 --padv 4 --padh 4 --alpha .4
```

The B0 detectors, downloaded from the release to `models/` on first use, are evaluated on the test split with 12001 images.
The results are written to `results/performance_b0.csv`.

```bash
python3 experiments/performance/evaluate_b0.py --method UERD --qf 98
```


## License

The code is licensed under the Mozilla Public License 2.0, see `LICENSE`.


## Citation

If you use this code, please cite

```bibtex
@inproceedings{benes2026steganalysis,
  title = {Steganalysis exploiting {JPEG} block padding},
  author = {Bene{\v{s}}, Martin and Hartmann, Friedrich and B{\"o}hme, Rainer},
  booktitle = {International Workshop on Information Forensics and Security (WIFS)},
  organization = {IEEE},
  pages = {1--6},
  year = {2026}
}
```

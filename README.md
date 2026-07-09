# Steganalysis Exploiting JPEG Block Padding

This is a source repository for the paper "Steganalysis Exploiting JPEG Block Padding", that is currently under peer-review at IEEE WIFS 2026.

Set it up with

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy scipy pillow jpeglib
```

For the submission, we provide the file `attack.py`, disclosing a simple proof-of-concept of the attack.
It can be used as follows

```bash
python3 attack.py \
    --precover "/home/username/cover.png" \
    --alpha .4 \
    --qf 98 \
    --pad 4 \
    --seed 12345
```

Upon acceptance, we plan to improve the structure of this repo to facilitate reproducibility of the results.

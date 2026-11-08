# B0 detectors

Pre-trained EfficientNet B0 detectors, one per steganographic method and JPEG quality, at 0.4 bpc.
Each `b0_<method>_q<qf>.pt` is a state dict, loaded by `experiments/performance/evaluate_b0.py`.
The weights (16 MB each) are attached to the [GitHub release](https://github.com/uibk-uncover/JPA/releases/latest), not part of the repository.
They are downloaded into this directory on first use.

| Model | Method | QF | Test P_E |
|---|---|---|---|
| `b0_EBS_q95.pt` | EBS | 95 | .073 |
| `b0_EBS_q98.pt` | EBS | 98 | .137 |
| `b0_JUNIWARD_q95.pt` | J-UNIWARD | 95 | .105 |
| `b0_JUNIWARD_q98.pt` | J-UNIWARD | 98 | .170 |
| `b0_UERD_q95.pt` | UERD | 95 | .123 |
| `b0_UERD_q98.pt` | UERD | 98 | .182 |
| `b0_nsF5_q95.pt` | nsF5 | 95 | .006 |
| `b0_nsF5_q98.pt` | nsF5 | 98 | .016 |

## Data

- Covers: ALASKA v2 grayscale 512x512 TIFFs, top-left cropped to 508x508 (padding 4x4), compressed with libjpeg 6b at the given QF.
- Stegos: simulated at 0.4 bpc, seeded by the SHA-256 of the image name (see `get_seed` in `experiments/performance/evaluate_b0.py`).
- Splits: 56003 training, 12001 validation, and 12001 test images out of 80005.
- Input: decompressed with libjpeg-turbo 2.1.0, scaled to [0, 1], no normalization, single channel.

## Training

| Parameter | Value |
|---|---|
| Network | timm `efficientnet_b0`, ImageNet pre-trained, no stem stride |
| Input | 1 channel, 508x508 |
| Loss | cross-entropy, 2 classes |
| Optimizer | AdamW, learning rate 1e-3 |
| Scheduler | one-cycle (warm-up 4 epochs, div. factor 25, final div. factor 10) |
| Epochs | 15 per curriculum step |
| Curriculum | 1.0 bpc, then 0.4 bpc, resumed from the 1.0 model |
| Batch size | 24, cover and stego not paired |
| Dropout | 0.1 |
| Augmentation | none |
| Precision | bfloat16 autocast in training |
| Model selection | lowest validation loss |

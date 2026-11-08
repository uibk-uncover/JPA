"""Evaluates a pre-trained EfficientNet B0 on cover-stego pairs.

Usage:
    python3 evaluate_b0.py --method UERD --qf 98
"""

import argparse
import hashlib
import jpeglib  #
import numpy as np  #
import os
import pandas as pd  #
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import timm  #
import torch  #
from tqdm import tqdm
from urllib.request import urlopen

sys.path.append(str(Path(__file__).resolve().parents[2] / 'data'))
from download import DATA_DIR, get_precover
from _stego import embed

MODEL_URL = 'https://github.com/uibk-uncover/JPA/releases/latest/download/{}'
MODEL_DIR = Path(__file__).resolve().parents[2] / 'models'


def get_model(path: Path) -> torch.nn.Module:
    """loads B0 without stem stride, for grayscale input, downloads the state dict from the release, unless cached"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen(MODEL_URL.format(path.name), timeout=60).read())
    model = timm.create_model('efficientnet_b0', in_chans=1, num_classes=2)
    model.conv_stem.stride = (1, 1)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def get_seed(name: str) -> int:
    """derives the stego seed from the image name, as in the B0 training data"""
    image_seed = int(hashlib.sha256(Path(name).stem.encode()).hexdigest(), 16) % 2**31
    r = np.random.default_rng(image_seed).integers(0, 2**64 - 1, dtype=np.uint64, size=4)
    return int.from_bytes(b''.join(int(v).to_bytes(8, 'big') for v in r), 'big')


def decompress_padded(path: str) -> np.ndarray:
    """decompresses the visible image, edge-padded to the block grid, for J-UNIWARD as in the B0 training data"""
    x = jpeglib.read_spatial(path).spatial[..., 0]
    return np.pad(x, ((0, -x.shape[0] % 8), (0, -x.shape[1] % 8)), mode='edge')


class StegoDataset(torch.utils.data.Dataset):
    """cover-stego pairs, compressed and embedded on the fly"""

    def __init__(self, images: pd.DataFrame, method: str, alpha: float, qf: int, pad: int, data_dir: Path):
        self.rows = images.to_dict('records')
        self.method, self.alpha, self.qf, self.pad, self.data_dir = method, alpha, qf, pad, data_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """returns the decompressed cover and stego, 2 1 H W, in [0, 1]"""
        row = self.rows[idx]
        with NamedTemporaryFile(suffix='.jpeg') as cover, NamedTemporaryFile(suffix='.jpeg') as stego:
            # compress
            x0 = get_precover(row['name'], self.data_dir, sha1=row.get('sha1'))[:-self.pad, :-self.pad]
            with jpeglib.version('turbo210'):
                jpeglib.from_spatial(x0[..., None]).write_spatial(cover.name, qt=self.qf)
            jpeg = jpeglib.read_dct(cover.name)

            # embed
            seed = get_seed(row['name'])
            jpeg.Y, _ = embed(self.method, y0=jpeg.Y, qt=jpeg.qt[jpeg.quant_tbl_no[0]], alpha=self.alpha, seed=seed, x0=decompress_padded(cover.name))
            jpeg.write_dct(stego.name)

            # decompress
            with jpeglib.version('turbo210'):
                x = np.stack([jpeglib.read_spatial(f.name).spatial for f in (cover, stego)])  # 2 H W 1
        return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation of a pre-trained B0.")
    parser.add_argument("--model", default=None, type=Path, help='path to the state dict, models/b0_<method>_q<qf>.pt by default')
    parser.add_argument("--images", default=Path(__file__).resolve().parents[2] / 'models' / 'split_te.csv', type=Path, help='CSV with test images (stem or name, optionally sha1)')
    parser.add_argument("--method", required=True, choices=['nsF5', 'EBS', 'JUNIWARD', 'UERD'], help='steganographic method')
    parser.add_argument("--alpha", default=.4, type=float, help='embedding rate [bits per coefficient]')
    parser.add_argument("--qf", default=98, type=int, help='JPEG quality factor')
    parser.add_argument("--pad", default=4, type=int, help='(square) pad size')
    parser.add_argument("--data", default=DATA_DIR, type=Path, help='cache for the precovers')
    parser.add_argument("--output", default=Path(__file__).resolve().parents[2] / 'results' / 'performance_b0.csv', type=Path, help='CSV with results')
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help='number of data loader workers')
    parser.add_argument("--batch_size", default=2, type=int, help='cover-stego pairs per batch')
    args = parser.parse_args()
    if args.model is None:
        args.model = MODEL_DIR / f'b0_{args.method}_q{args.qf}.pt'

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = get_model(args.model).to(device)

    # data
    images = pd.read_csv(args.images, dtype=str).rename(columns={'stem': 'name'})
    dataset = StegoDataset(images, method=args.method, alpha=args.alpha, qf=args.qf, pad=args.pad, data_dir=args.data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # detect
    scores = []  # N 2, cover and stego
    with torch.no_grad():
        for x in tqdm(loader):
            y_hat = torch.softmax(model(x.flatten(0, 1).to(device)), dim=1)[:, 1]  # 2B
            scores.extend(y_hat.cpu().numpy().reshape(-1, 2))

    # P_E, minimized over thresholds
    s0, s1 = np.array(scores).T
    taus = np.r_[np.unique(scores), np.inf]
    pe = min(.5 * ((s0 >= t).mean() + (s1 < t).mean()) for t in taus)

    # write the repo column of this setting, next to the paper column
    key = {'stego_method': args.method, 'alpha': args.alpha, 'qf': args.qf, 'padv': args.pad, 'padh': args.pad}
    results = pd.read_csv(args.output)
    is_setting = (results[list(key)] == pd.Series(key)).all(axis=1)
    assert is_setting.any(), f'setting {key} not in {args.output}'
    results.loc[is_setting, 'pe_b0_repo'] = pe
    results.to_csv(args.output, index=False)
    print(results[is_setting].to_string(index=False))

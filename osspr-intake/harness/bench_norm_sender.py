"""Baseline timing for ``norm_sender_normalized``.

NRQM's global GSM features call this once per image. Neighbor orientations
are appended with one ``cat`` per band. This script only measures the current
implementation; pyramid construction is not included.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pyiqa.archs.nrqm_arch import norm_sender_normalized
from pyiqa.matlab_utils.scfpyr_util import SCFpyr_PyTorch
from pyiqa.utils import imread2tensor
from pyiqa.utils.color_util import to_y_channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[2]
    path = root / 'tests/test_efficiency_img_dir/frame_9ogwhvvy7gfj6jzh.jpg'
    image = to_y_channel(imread2tensor(str(path)).unsqueeze(0), 255).double()
    pyramid = SCFpyr_PyTorch(height=2, nbands=6, device=image.device).build(image)
    bands = [level[..., 0] for level in pyramid[1]] + [
        level[..., 0] for level in pyramid[2]
    ]

    def once():
        with torch.inference_mode():
            return norm_sender_normalized(bands)

    for _ in range(args.warmup):
        once()
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        once()
        times.append((time.perf_counter() - start) * 1000.0)
    median = sorted(times)[len(times) // 2]
    noise = (max(times) - min(times)) / median * 100.0
    print(
        json.dumps(
            {
                'function': 'norm_sender_normalized',
                'baseline_median_ms': median,
                'baseline_noise_pct': noise,
                'times_ms': times,
                'seed': args.seed,
                'runs': args.runs,
            }
        )
    )


if __name__ == '__main__':
    main()

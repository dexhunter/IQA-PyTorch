"""Baseline timing for one ``gaborconvolve`` call.

MAD calls this twice per image pair. The filter bank is rebuilt with NumPy
and applied as 20 separate inverse FFTs. This script only measures that path.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pyiqa.archs.mad_arch import gaborconvolve
from pyiqa.utils import imread2tensor


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
    # MAD.mad scales RGB to 255 before lo_index -> gaborconvolve.
    image = imread2tensor(str(path)).unsqueeze(0) * 255.0

    def once():
        with torch.inference_mode():
            return gaborconvolve(image)

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
                'function': 'gaborconvolve',
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

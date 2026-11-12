"""Baseline timing for one ``VIF.vifsub_est_M`` call.

The subband windows are flat boxes, but each statistic is a dense 2D
convolution. Pyramid construction is excluded so the number is the cost of
this method alone.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pyiqa.archs.vif_arch import VIF, SteerablePyramidSpace
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
    reference = imread2tensor(str(path)).unsqueeze(0)
    distorted = torch.clamp(reference * 0.85 + 0.08, 0, 1)
    metric = VIF()
    reference_y = to_y_channel(reference, 255)
    distorted_y = to_y_channel(distorted, 255)
    reference_pyr = SteerablePyramidSpace(
        reference_y,
        height=metric.level,
        order=metric.ori,
        channels=metric.channels,
    )[::-1]
    distorted_pyr = SteerablePyramidSpace(
        distorted_y,
        height=metric.level,
        order=metric.ori,
        channels=metric.channels,
    )[::-1]

    def once():
        with torch.inference_mode():
            return metric.vifsub_est_M(reference_pyr, distorted_pyr)

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
                'function': 'VIF.vifsub_est_M',
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

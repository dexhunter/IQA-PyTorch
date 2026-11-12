"""Baseline timing for ``ilniqe`` on one real photograph.

The Gaussian-derivative filters inside this call are rank-1, but each of the
100 convolutions still uses a full 2D kernel. This script only measures the
current implementation.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pyiqa.archs.func_util import diff_round
from pyiqa.archs.niqe_arch import ilniqe
from pyiqa.utils import imread2tensor


def _load_image(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    root = Path(__file__).resolve().parents[2]
    path = root / 'tests/test_efficiency_img_dir/frame_9ogwhvvy7gfj6jzh.jpg'
    return imread2tensor(str(path)).unsqueeze(0)


def _prepare(seed: int):
    import pyiqa

    image = _load_image(seed)
    metric = pyiqa.create_metric('ilniqe', device='cpu')
    net = metric.net
    prepared = diff_round(image * 255.0).to(torch.float64)
    batch = prepared.shape[0]
    return (
        prepared,
        net.mu_pris_param.to(prepared).repeat(batch, 1),
        net.cov_pris_param.to(prepared).repeat(batch, 1, 1),
        net.principleVectors.to(prepared).repeat(batch, 1, 1),
        net.meanOfSampleData.to(prepared).repeat(batch, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    prepared, mu, cov, vectors, mean = _prepare(args.seed)

    def once():
        with torch.inference_mode():
            return ilniqe(prepared, mu, cov, vectors, mean)

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
                'function': 'ilniqe',
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

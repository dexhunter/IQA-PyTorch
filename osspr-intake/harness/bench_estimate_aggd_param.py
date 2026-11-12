"""Baseline timing for one large ``estimate_aggd_param`` call.

ILNIQE fits an AGGD on every log-Gabor channel of every 84x84 block. For a
photo resized to the metric's 524px working size that is 36 x 78 blocks of
84x84. The current lookup broadcasts a 9801-entry gamma table across that
batch. This script only measures that call.
"""

import argparse
import json
import time

import torch

from pyiqa.archs.func_util import estimate_aggd_param


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    block = torch.randn(36 * 78, 1, 84, 84)

    def once():
        with torch.inference_mode():
            return estimate_aggd_param(block)

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
                'function': 'estimate_aggd_param',
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

"""Wall-time baseline for pyiqa.archs.mad_arch.ical_stat.

The timed call is one local-moment map on a single Gabor magnitude band.
MAD.lo_index builds those bands with gaborconvolve and calls ical_stat once
per orientation, scale, and image (40 calls per RGB pair at the defaults).
Setup (the forward FFT filter bank) is outside the timed section.

baseline_noise_pct is the full spread (max - min) / median * 100.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import time

import torch

torch.set_num_threads(4)

from pyiqa.archs.mad_arch import gaborconvolve, ical_stat
from pyiqa.matlab_utils import math_util
from pyiqa.utils.img_util import imread2tensor

RUNS = 5
WARMUP = 1
IMAGE = 'tests/test_efficiency_img_dir/frame_4wrwr48cdk6rbcjl.jpg'


def main() -> None:
    # Same scale MAD.mad applies before lo_index (test_y_channel is false).
    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0) * 255.0
    band = math_util.abs(gaborconvolve(image)[0][0])

    def once():
        return ical_stat(band)

    for _ in range(WARMUP):
        once()

    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        std, skew, kurt = once()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if std.shape != skew.shape or std.shape != kurt.shape:
            raise SystemExit('ical_stat returned inconsistent maps')
        samples.append(elapsed_ms)

    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (max(samples) - min(samples)) / median * 100.0
    print('samples_ms=' + ','.join(f'{v:.4f}' for v in samples))
    print(f'median_ms={median:.4f}')
    print(f'noise_pct={noise:.6f}')
    print(f'band_shape={tuple(band.shape)} dtype={band.dtype}')


if __name__ == '__main__':
    main()

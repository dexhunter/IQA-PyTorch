"""Wall-time baseline for pyiqa.archs.mad_arch.gaborconvolve.

One call on a real RGB photo scaled to 0-255, which is the tensor MAD.mad
passes into lo_index. lo_index runs the filter bank twice (reference and
distorted). The numpy frequency grids and twenty sequential complex IFFTs
are inside the timed call.

baseline_noise_pct is the full spread (max - min) / median * 100.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import time

import torch

torch.set_num_threads(4)

from pyiqa.archs.mad_arch import gaborconvolve
from pyiqa.utils.img_util import imread2tensor

RUNS = 5
WARMUP = 1
IMAGE = 'tests/test_efficiency_img_dir/frame_4wrwr48cdk6rbcjl.jpg'


def main() -> None:
    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0) * 255.0

    def once():
        return gaborconvolve(image)

    for _ in range(WARMUP):
        once()

    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        bands = once()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if len(bands) != 4 or len(bands[0]) != 5:
            raise SystemExit('gaborconvolve did not return 4 orientations x 5 scales')
        samples.append(elapsed_ms)

    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (max(samples) - min(samples)) / median * 100.0
    print('samples_ms=' + ','.join(f'{v:.4f}' for v in samples))
    print(f'median_ms={median:.4f}')
    print(f'noise_pct={noise:.6f}')
    print(f'image_shape={tuple(image.shape)}')


if __name__ == '__main__':
    main()

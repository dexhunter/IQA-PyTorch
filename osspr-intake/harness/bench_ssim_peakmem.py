"""Peak-memory baseline for pyiqa.archs.ssim_arch.ssim.

The workload is a batch of 4 copies of the RGB photo shipped with the tests,
converted the way SSIM.forward does (luminance, float64, data_range 255,
downsample false) before the call. One copy is only a few allocator buckets,
so its high-water growth bounces by about 19 percent; the batch of 4 makes
the five-plane moment stack the growth we record.

Each sample is a fresh process. The value is the increase in ru_maxrss across
the ssim call, in bytes. baseline_noise_pct is that full spread
(max - min) / median * 100.
"""

import os
import resource
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

RUNS = 5
IMAGE = 'tests/test_efficiency_img_dir/frame_4wrwr48cdk6rbcjl.jpg'
BATCH = 4


def worker() -> None:
    import gc

    import torch

    torch.set_num_threads(4)
    from pyiqa.archs.func_util import preprocess_rgb
    from pyiqa.archs.ssim_arch import ssim
    from pyiqa.utils.img_util import imread2tensor

    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0).repeat(BATCH, 1, 1, 1)
    reference = preprocess_rgb(image, True, 255, 'yiq').to(torch.float64)
    distorted = preprocess_rgb(image.clone(), True, 255, 'yiq').to(torch.float64)
    del image
    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = ssim(reference, distorted, data_range=255, downsample=False)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not torch.isfinite(score).all():
        raise SystemExit('ssim returned a non-finite score')
    print((after_kb - before_kb) * 1024)


def main() -> None:
    if '--worker' in sys.argv:
        worker()
        return

    samples = []
    for _ in range(RUNS):
        out = subprocess.check_output(
            [sys.executable, __file__, '--worker'],
            env=os.environ.copy(),
            text=True,
        )
        samples.append(int(out.strip().splitlines()[-1]))

    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (max(samples) - min(samples)) / median * 100.0
    print('samples_bytes=' + ','.join(str(v) for v in samples))
    print(f'median_bytes={median}')
    print(f'noise_pct={noise:.4f}')


if __name__ == '__main__':
    main()

"""Peak-memory baseline for pyiqa.archs.mad_arch.lo_index.

One lo_index() call on the RGB photo shipped with the tests, scaled by 255,
which is the tensor MAD.mad passes when it is not in luminance-only mode.
Reference and distorted inputs are that same photo.

Each sample is a fresh process. The value is the increase in ru_maxrss across
the lo_index call, in bytes. baseline_noise_pct is that full spread
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


def worker() -> None:
    import gc

    import torch

    torch.set_num_threads(4)
    from pyiqa.archs.mad_arch import lo_index
    from pyiqa.utils.img_util import imread2tensor

    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0) * 255
    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = lo_index(image, image)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not torch.isfinite(score).all():
        raise SystemExit('lo_index returned a non-finite score')
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

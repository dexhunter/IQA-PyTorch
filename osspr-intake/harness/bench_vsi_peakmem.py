"""Peak-memory baseline for pyiqa.archs.vsi_arch.vsi.

One vsi() call on the RGB photo shipped with the tests, values in [0, 1],
data_range 1, which is what VSI.forward passes through.

Each sample is a fresh process. The value is the increase in ru_maxrss across
the vsi call, in bytes. baseline_noise_pct is that full spread
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
    from pyiqa.archs.vsi_arch import vsi
    from pyiqa.utils.img_util import imread2tensor

    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0)
    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = vsi(image, image, data_range=1.0)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not torch.isfinite(score).all():
        raise SystemExit('vsi returned a non-finite score')
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

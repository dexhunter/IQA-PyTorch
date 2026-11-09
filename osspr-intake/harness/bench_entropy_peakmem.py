"""Peak-memory baseline for pyiqa.archs.entropy_arch.entropy.

Runs the metric on a real RGB photo shipped with the tests. Each sample is a
fresh process so ru_maxrss is the high-water mark of that process. The
reported value is the increase in that high-water mark across the entropy
call, in bytes.

baseline_noise_pct is the full spread (max - min) / median * 100 over the
samples.
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
    from pyiqa.archs.entropy_arch import entropy
    from pyiqa.utils.img_util import imread2tensor

    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0)
    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = entropy(image)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Keep the result live until the high-water mark is read.
    if not torch.isfinite(score).all():
        raise SystemExit('entropy returned a non-finite score')
    print((after_kb - before_kb) * 1024)


def main() -> None:
    if '--worker' in sys.argv:
        worker()
        return

    samples = []
    for _ in range(RUNS):
        env = os.environ.copy()
        out = subprocess.check_output(
            [sys.executable, __file__, '--worker'],
            env=env,
            text=True,
        )
        samples.append(int(out.strip().splitlines()[-1]))

    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (max(samples) - min(samples)) / median * 100.0
    print('samples_bytes=' + ','.join(str(v) for v in samples))
    print(f'median_bytes={median}')
    print(f'noise_pct={noise:.6f}')


if __name__ == '__main__':
    main()

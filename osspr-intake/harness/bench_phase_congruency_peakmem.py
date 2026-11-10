"""Peak-memory baseline for pyiqa.archs.fsim_arch._phase_congruency.

The input is a batch of 4 copies of the luminance map fsim() builds for the
RGB photo shipped with the tests: scale to 0-255, average-pool by
round(min(H, W) / 256), then take the Y channel. Defaults are fsim's
(4 scales, 4 orientations). One copy's high-water growth is about 53 MiB and
moved by a few percent across allocator buckets; the batch of 4 makes the
16-filter complex bank the growth we record.

Each sample is a fresh process. The value is the increase in ru_maxrss across
the _phase_congruency call, in bytes. baseline_noise_pct is that full spread
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
    from pyiqa.archs.fsim_arch import _phase_congruency
    from pyiqa.utils.color_util import rgb2yiq
    from pyiqa.utils.img_util import imread2tensor

    image = imread2tensor(IMAGE, rgb=True).unsqueeze(0).repeat(BATCH, 1, 1, 1)
    kernel = max(1, round(min(image.shape[-2:]) / 256))
    pooled = torch.nn.functional.avg_pool2d(image * 255, kernel)
    luma = rgb2yiq(pooled)[:, :1]
    del image, pooled
    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = _phase_congruency(luma)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not torch.isfinite(score).all():
        raise SystemExit('_phase_congruency returned a non-finite map')
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

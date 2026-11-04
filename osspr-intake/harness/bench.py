#!/usr/bin/env python3
"""Median-time benchmark for pyiqa.archs.mad_arch.get_moments.

Inputs match the tensor MAD actually passes in: 16x16 patches with stride 4
taken from 3x384x512 frames, the resolution of ResultsCalibra (the images the
MAD calibration table uses).
The timed call uses sk=True, which is 40 of the 42 get_moments calls inside
one MAD forward (lo_index, 4 orientations x 5 scales x ref/dst).

A batch of four frames matches a small evaluation batch of those calibration
images. One frame is only ~70 ms and the median wanders by more than 5% across
process restarts on this VM; four frames keeps the same patch geometry while
the median settles.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import statistics
import time

import torch

from pyiqa.archs.mad_arch import extract_patches_2d, get_moments

# ResultsCalibra BMP size is 512x384. MAD's ical_stat / ical_std defaults.
IMAGE_SHAPE = (4, 3, 384, 512)
PATCH = 16
STRIDE = 4
WARMUP = 10
REPEATS = 20
SEED = 0


def make_patches() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    image = torch.rand(*IMAGE_SHAPE, generator=generator, dtype=torch.float32) * 255.0
    return extract_patches_2d(image, patch_shape=[PATCH, PATCH], step=[STRIDE, STRIDE])


def main() -> None:
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    patches = make_patches()
    if patches.device.type != "cpu":
        raise RuntimeError(f"benchmark expects CPU tensors, got {patches.device}")

    with torch.no_grad():
        mean, std, skews, kurt = get_moments(patches, sk=True)
        if mean.shape != std.shape or skews.shape != kurt.shape:
            raise RuntimeError("unexpected get_moments shapes")
        if not torch.isfinite(std).all():
            raise RuntimeError("non-finite std from get_moments")

        for _ in range(WARMUP):
            get_moments(patches, sk=True)

        samples_ms: list[float] = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            get_moments(patches, sk=True)
            samples_ms.append((time.perf_counter() - start) * 1e3)

    median_ms = statistics.median(samples_ms)
    print(f"input_shape={tuple(patches.shape)}")
    print(f"sk=True")
    print(f"warmup={WARMUP}")
    print(f"repeats={REPEATS}")
    print(f"min_ms={min(samples_ms):.6f}")
    print(f"max_ms={max(samples_ms):.6f}")
    print(f"median_ms={median_ms:.6f}")


if __name__ == "__main__":
    main()

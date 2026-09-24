"""Surrogate timing under the same conditions as the solver.

Wall-clock timing including preprocessing and model loading, with the
sensitivity to batch size. This script reports, on one CPU core:

  * one-time model-load cost (paid once, then amortised over every query);
  * single-design latency, warm model, INCLUDING input normalisation -- the
    representative single-design engineering use; and
  * per-design cost inside batches of increasing size -- the many-query use
    that motivates the surrogate.

The reference-solver per-design time is measured separately by
scripts/timeSolver.ts on the same machine.

Run:  python timing_surrogate.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet

ARCH = "deepBeam"
BATCHES = [1, 8, 64, 512, 4096]
REPEATS = 50


def main() -> None:
    cfg = get_config(archetype=ARCH)
    data = load_archetype(cfg)
    mean = data.theta_mean.numpy()
    std = data.theta_std.numpy()
    # raw (un-normalised) design parameters, as a user would supply them
    raw = (data.theta * data.theta_std + data.theta_mean).numpy()
    torch.set_num_threads(1)

    # ---- one-time model-load cost ----------------------------------------
    t0 = time.perf_counter()
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    model.load_state_dict(torch.load(f"runs/{ARCH}/model.pt",
                          map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1e3

    # ---- single-design latency, warm, incl. normalisation ----------------
    x1 = raw[:1]
    with torch.no_grad():
        for _ in range(10):  # warm-up
            t = torch.tensor((x1 - mean) / std, dtype=torch.float32)
            _ = model(t)
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            tt = torch.tensor((x1 - mean) / std, dtype=torch.float32)
            _ = model(tt)
            ts.append(time.perf_counter() - t0)
    single_us = np.median(ts) * 1e6

    print(f"model-load (one-time)              : {load_ms:8.2f} ms")
    print(f"single-design latency (warm, +norm): {single_us:8.2f} us\n")
    print(f"{'batch':>8s} {'per-design (us)':>16s} {'throughput (1/s)':>18s}")
    n = raw.shape[0]
    for B in BATCHES:
        xb = raw[np.random.randint(0, n, size=B)]
        with torch.no_grad():
            for _ in range(5):
                tb = torch.tensor((xb - mean) / std, dtype=torch.float32)
                _ = model(tb)
            ts = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                tb = torch.tensor((xb - mean) / std, dtype=torch.float32)
                _ = model(tb)
                ts.append(time.perf_counter() - t0)
        per = np.median(ts) / B * 1e6
        print(f"{B:8d} {per:16.3f} {1e6 / per:18.0f}")


if __name__ == "__main__":
    main()

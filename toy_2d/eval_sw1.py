"""Step 16: sliced Wasserstein-1 for 1-RF and 2-RF at 1, 2, 5, 20 Euler steps.
Run from the repository root:  python toy_2d/eval_sw1.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons

SW_DIRECTIONS = 512   # number of random projection directions
SW_SEED = 0           # fixed seed -> same directions for every comparison


def sliced_w1(a, b, n_dirs=SW_DIRECTIONS, seed=SW_SEED):
    """Sliced Wasserstein-1 between two equal-size 2D point clouds."""
    assert a.shape == b.shape and a.shape[1] == 2
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)   # unit directions
    proj_a = np.sort(a @ dirs.T, axis=0)                  # (N, n_dirs), sorted per direction
    proj_b = np.sort(b @ dirs.T, axis=0)
    return float(np.abs(proj_a - proj_b).mean())          # mean over points and directions


def main():
    d1 = np.load("toy_2d/runs/1rf/eval_outputs.npz")
    d2 = np.load("toy_2d/runs/2rf/eval_outputs.npz")
    ref = d1["reference_x1"]
    assert np.array_equal(d1["x0"], d2["x0"]), "models must share starting points"
    assert np.array_equal(ref, d2["reference_x1"]), "models must share the reference"

    fresh, _ = make_moons(n_samples=len(ref), noise=0.05, random_state=303)
    fresh = fresh.astype(np.float32)
    floor, ceiling = sliced_w1(ref, fresh), sliced_w1(ref, d1["x0"])
    print("self-check  SW(ref, ref)            =", sliced_w1(ref, ref))
    print(f"floor       SW(ref, fresh moons)    = {floor:.4f}")
    print(f"ceiling     SW(ref, Gaussian noise) = {ceiling:.4f}")

    rows = []
    for steps in (1, 2, 5, 20):
        key = f"euler_{steps:02d}"
        rows.append({"steps": steps,
                     "sw1_1rf": sliced_w1(d1[key], ref),
                     "sw1_2rf": sliced_w1(d2[key], ref)})

    print(f"\n{'steps':>5} | {'1-RF SW1':>9} | {'2-RF SW1':>9} | {'1-RF / 2-RF':>11}")
    for r in rows:
        print(f"{r['steps']:>5} | {r['sw1_1rf']:9.4f} | {r['sw1_2rf']:9.4f} | "
              f"{r['sw1_1rf'] / r['sw1_2rf']:11.2f}")

    out = Path("toy_2d/runs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sw1.json").write_text(json.dumps({
        "metric": "sliced Wasserstein-1", "directions": SW_DIRECTIONS,
        "projection_seed": SW_SEED, "num_points": len(ref),
        "floor_fresh_moons_seed": 303, "floor": floor, "ceiling_gaussian": ceiling,
        "rows": rows}, indent=2) + "\n")
    print("\nsaved toy_2d/runs/metrics/sw1.json")


if __name__ == "__main__":
    main()

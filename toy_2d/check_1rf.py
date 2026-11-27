"""Step 11: smoke-test the saved 1-RF checkpoint.

Loads the checkpoint on CPU, takes ONE Euler step from the first 8
evaluation Gaussian points, and compares the result with the saved
euler_01 rows in eval_outputs.npz.

Run from the repository root:
    python -m toy_2d.check_1rf
"""
import argparse
import platform
from pathlib import Path

import numpy as np
import torch

from .euler import sample_euler
from .rectified_flow import VelocityMLP

RUNS = Path(__file__).resolve().parent / "runs"
TOLERANCE = 1e-6  # float32 agreement threshold (max absolute difference)


def load_velocity(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VelocityMLP(width=checkpoint["settings"]["model"]["width"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["settings"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUNS / "1rf")
    args = parser.parse_args()

    print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | {platform.platform()}")

    model, settings = load_velocity(args.run / "checkpoint.pt")
    print(f"Loaded 1-RF: width={settings['model']['width']}, "
          f"updates={settings['training']['updates']}, trained with torch {settings['torch_version']}")

    with np.load(args.run / "eval_outputs.npz", allow_pickle=False) as data:
        print("Archive keys:", sorted(data.files))
        x0 = torch.from_numpy(data["x0"][:8])
        saved = data["euler_01"][:8]

    endpoints = sample_euler(model, x0, steps=1).numpy()

    print("Output shape:", endpoints.shape)
    print("All finite:  ", bool(np.isfinite(endpoints).all()))
    max_diff = float(np.abs(endpoints - saved).max())
    print(f"Max |new - saved euler_01|: {max_diff:.3e}  (tolerance {TOLERANCE:g})")

    assert endpoints.shape == (8, 2), "wrong shape"
    assert np.isfinite(endpoints).all(), "non-finite values"
    assert max_diff <= TOLERANCE, "new endpoints do not match the saved euler_01 rows"
    print("PASS: 1-RF checkpoint loads and reproduces the saved one-step outputs.")


if __name__ == "__main__":
    main()

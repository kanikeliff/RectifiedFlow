"""Step 17: trajectory straightness (paper Eq. 3) for 1-RF and 2-RF.
Run from the repository root:  python -m toy_2d.eval_straightness
"""
import json
from pathlib import Path

import numpy as np
import torch

from .check_1rf import load_velocity
from .euler import sample_euler

FINE_STEPS = 100   # same fine Euler resolution for both models


def straightness(model, x0, steps=FINE_STEPS):
    """Discrete estimate of S(Z) = E ∫ ||(Z1 - Z0) - dZt/dt||^2 dt."""
    traj = sample_euler(model, x0, steps, return_trajectory=True)   # (steps+1, N, 2)
    displacement = traj[-1] - traj[0]                               # Z_end - Z_start, (N, 2)
    step_velocity = (traj[1:] - traj[:-1]) * steps                  # (Z_{k+1}-Z_k)/h, (steps, N, 2)
    sq_dev = ((step_velocity - displacement) ** 2).sum(dim=-1)      # (steps, N)
    s = sq_dev.mean().item()                                        # mean over time and points
    rel = s / (displacement ** 2).sum(dim=-1).mean().item()         # scale-free version
    return s, rel, traj


def main():
    torch.set_num_threads(2)
    x0 = torch.from_numpy(np.load("toy_2d/runs/1rf/eval_outputs.npz")["x0"])
    results, trajectories = {}, {}
    for name in ("1rf", "2rf"):
        model, _ = load_velocity(f"toy_2d/runs/{name}/checkpoint.pt")
        s, rel, traj = straightness(model, x0)
        results[name] = {"S": s, "S_relative": rel}
        trajectories[name] = traj.numpy()
        print(f"{name.upper()}: S = {s:.4f}   relative S = {rel:.4f}")
    print(f"ratio S(1-RF) / S(2-RF) = {results['1rf']['S'] / results['2rf']['S']:.1f}")

    out = Path("toy_2d/runs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trajectories.npz", x0=x0.numpy(), **trajectories)
    (out / "straightness.json").write_text(json.dumps({
        "estimator": "mean over 100 Euler steps and 2048 eval points of ||v_k - (z_end - z_start)||^2",
        "fine_steps": FINE_STEPS, "num_points": len(x0), "results": results}, indent=2) + "\n")
    print("saved toy_2d/runs/metrics/straightness.json and trajectories.npz")


if __name__ == "__main__":
    main()

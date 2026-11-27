"""Steps 12-15: reflow pairs from the 1-RF teacher, 2-RF training, and 2-RF outputs.

Run from the repository root, after train_1rf:
    python -m toy_2d.train_2rf
"""
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .check_1rf import load_velocity
from .euler import sample_euler
from .rectified_flow import VelocityMLP, fit_velocity
from .train_1rf import EULER_STEPS, MODEL_SEED, MODEL_WIDTH, TRAINING

RUNS = Path(__file__).resolve().parent / "runs"
TEACHER_STEPS = 200          # chosen from the 100-vs-200 gap check below
CHECK_STEPS = (100, 200)
CHECK_SUBSET = 512           # first rows of the training split


def main():
    torch.set_num_threads(2)
    teacher, teacher_settings = load_velocity(RUNS / "1rf" / "checkpoint.pt")
    with np.load(RUNS / "data" / "train.npz", allow_pickle=False) as data:
        x0_train = torch.from_numpy(data["x0"])     # training Gaussians only

    # Step 12: teacher step count check
    coarse = sample_euler(teacher, x0_train[:CHECK_SUBSET], CHECK_STEPS[0])
    fine = sample_euler(teacher, x0_train[:CHECK_SUBSET], CHECK_STEPS[1])
    gap = (coarse - fine).norm(dim=1)
    gap_stats = {"mean": gap.mean().item(), "median": gap.median().item(),
                 "p95": gap.quantile(0.95).item(), "max": gap.max().item()}
    print(f"Teacher gap {CHECK_STEPS[0]} vs {CHECK_STEPS[1]} steps:",
          {k: round(v, 4) for k, v in gap_stats.items()})

    # Step 13: reflow pairs, row i of x1_hat comes from row i of x0_train
    x1_hat = sample_euler(teacher, x0_train, TEACHER_STEPS)
    if not torch.isfinite(x1_hat).all():
        raise RuntimeError("Teacher produced non-finite endpoints.")
    check = sample_euler(teacher, x0_train[:8], TEACHER_STEPS)
    same_rows = (check - x1_hat[:8]).abs().max().item()
    shifted_rows = (check - x1_hat[1:9]).abs().max().item()
    print(f"Pairing check: same rows {same_rows:.2e}, shifted rows {shifted_rows:.2f}")
    if same_rows > 1e-5:
        raise RuntimeError("Reflow pairs lost their row alignment.")

    reflow_dir = RUNS / "reflow"
    reflow_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(reflow_dir / "pairs.npz", x0=x0_train.numpy(), x1=x1_hat.numpy())
    reflow_settings = {
        "teacher_checkpoint": "toy_2d/runs/1rf/checkpoint.pt",
        "teacher_steps": TEACHER_STEPS,
        "teacher_step_check": {"subset_size": CHECK_SUBSET, "compared_steps": list(CHECK_STEPS),
                               "gap": gap_stats},
        "pairing_check": {"same_rows_max_diff": same_rows, "shifted_rows_max_diff": shifted_rows},
        "source": "training split only (runs/data/train.npz x0); eval points not used",
        "num_pairs": len(x0_train),
        "data": teacher_settings["data"],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    (reflow_dir / "settings.json").write_text(json.dumps(reflow_settings, indent=2) + "\n")

    # Step 14: 2-RF with the same architecture, optimizer settings, updates, and seeds as 1-RF
    torch.manual_seed(MODEL_SEED)
    model = VelocityMLP(width=MODEL_WIDTH)
    print(f"Training 2-RF on {len(x0_train)} reflow pairs for {TRAINING['updates']} updates.",
          flush=True)
    losses = fit_velocity(model, x0_train, x1_hat, **TRAINING)
    if not np.isfinite(losses).all():
        raise RuntimeError("Training produced a non-finite loss.")

    out = RUNS / "2rf"
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "model": {"width": MODEL_WIDTH, "seed": MODEL_SEED},
        "training": TRAINING,
        "pairs": "toy_2d/runs/reflow/pairs.npz",
        "reflow": reflow_settings,
        "euler_steps": list(EULER_STEPS),
        "device": "cpu",
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
    }
    torch.save({"settings": settings, "model_state": model.state_dict()}, out / "checkpoint.pt")
    np.save(out / "training_loss.npy", np.asarray(losses))
    loss_1rf = np.load(RUNS / "1rf" / "training_loss.npy")
    print(f"Final loss (mean of last 100): 1-RF {loss_1rf[-100:].mean():.4f}, "
          f"2-RF {np.mean(losses[-100:]):.4f}")

    restored, _ = load_velocity(out / "checkpoint.pt")
    model.eval()
    with np.load(RUNS / "1rf" / "eval_outputs.npz", allow_pickle=False) as data:
        eval_x0 = torch.from_numpy(data["x0"])      # same starting points as 1-RF
        reference_x1 = data["reference_x1"]
    with torch.no_grad():
        probe = eval_x0[:8]
        t = probe.new_full((len(probe), 1), 0.5)
        torch.testing.assert_close(restored(probe, t), model(probe, t), rtol=0, atol=0)
    print("Checkpoint reload verified on 8 held-out Gaussian points.")

    # Step 15: 2-RF outputs at the evaluation step budgets
    outputs = {"x0": eval_x0.numpy(), "reference_x1": reference_x1}
    for steps in EULER_STEPS:
        endpoints = sample_euler(restored, eval_x0, steps)
        if endpoints.shape != eval_x0.shape or not torch.isfinite(endpoints).all():
            raise RuntimeError(f"Invalid endpoints for {steps} Euler steps.")
        outputs[f"euler_{steps:02d}"] = endpoints.numpy()
        print(f"Euler {steps:2d} steps → {tuple(endpoints.shape)} finite endpoints.")
    np.savez_compressed(out / "eval_outputs.npz", **outputs)
    print(f"Saved reflow pairs to {reflow_dir} and 2-RF outputs to {out}")


if __name__ == "__main__":
    main()

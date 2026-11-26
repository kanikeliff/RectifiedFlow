import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .euler import sample_euler
from .rectified_flow import VelocityMLP, fit_velocity


RUNS = Path(__file__).resolve().parent / "runs"
EULER_STEPS = (1, 2, 5, 20)
MODEL_WIDTH = 128
MODEL_SEED = 42
TRAINING = {
    "updates": 3000,
    "batch_size": 256,
    "lr": 1e-3,
    "seed": 43,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUNS / "1rf")
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(MODEL_SEED)

    with np.load(RUNS / "data" / "train.npz", allow_pickle=False) as data:
        train_x0 = torch.from_numpy(data["x0"])
        train_x1 = torch.from_numpy(data["x1"])

    model = VelocityMLP(width=MODEL_WIDTH)
    print(f"Training 1-RF on {len(train_x0)} pairs for {TRAINING['updates']} updates.", flush=True)
    losses = fit_velocity(model, train_x0, train_x1, **TRAINING)
    if not np.isfinite(losses).all():
        raise RuntimeError("Training produced a non-finite loss.")

    args.output.mkdir(parents=True, exist_ok=True)
    data_settings = json.loads(
        (RUNS / "data" / "settings.json").read_text(encoding="utf-8")
    )
    settings = {
        "model": {"width": MODEL_WIDTH, "seed": MODEL_SEED},
        "training": TRAINING,
        "data": data_settings,
        "euler_steps": list(EULER_STEPS),
        "device": "cpu",
        "torch_version": str(torch.__version__),
    }

    checkpoint_path = args.output / "checkpoint.pt"
    torch.save({"settings": settings, "model_state": model.state_dict()}, checkpoint_path)
    np.save(args.output / "training_loss.npy", np.asarray(losses))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = VelocityMLP(width=checkpoint["settings"]["model"]["width"])
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval()
    model.eval()

    with np.load(RUNS / "data" / "eval.npz", allow_pickle=False) as data:
        eval_x0 = torch.from_numpy(data["x0"])
        reference_x1 = data["x1"]

    with torch.no_grad():
        probe = eval_x0[:8]
        t = probe.new_full((len(probe), 1), 0.5)
        torch.testing.assert_close(restored(probe, t), model(probe, t), rtol=0, atol=0)
    print("Checkpoint reload verified on 8 held-out Gaussian points.")

    outputs = {"x0": eval_x0.numpy(), "reference_x1": reference_x1}
    for steps in EULER_STEPS:
        endpoints = sample_euler(restored, eval_x0, steps)
        if endpoints.shape != eval_x0.shape or not torch.isfinite(endpoints).all():
            raise RuntimeError(f"Invalid endpoints for {steps} Euler steps.")
        outputs[f"euler_{steps:02d}"] = endpoints.numpy()
        print(f"Euler {steps:2d} steps → {tuple(endpoints.shape)} finite endpoints.")

    np.savez_compressed(args.output / "eval_outputs.npz", **outputs)
    print(f"Saved checkpoint, training loss, and evaluation outputs to {args.output}")


if __name__ == "__main__":
    main()

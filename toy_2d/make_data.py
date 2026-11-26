import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


OUTPUT = Path(__file__).resolve().parent / "runs" / "data"
MOONS_NOISE = 0.05
SPLITS = {
    "train": {"count": 8192, "gaussian_seed": 101, "moons_seed": 102},
    "eval": {"count": 2048, "gaussian_seed": 201, "moons_seed": 202},
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")

    for row, (split, settings) in enumerate(SPLITS.items()):
        count = settings["count"]
        rng = np.random.default_rng(settings["gaussian_seed"])
        x0 = rng.standard_normal((count, 2)).astype(np.float32)

        x1, _ = make_moons(
            n_samples=count,
            noise=MOONS_NOISE,
            random_state=settings["moons_seed"],
        )
        x1 = x1.astype(np.float32)

        np.savez_compressed(OUTPUT / f"{split}.npz", x0=x0, x1=x1)
        print(f"{split}: x0={x0.shape}, x1={x1.shape}, dtype={x0.dtype}")

        for axis, points, name in zip(
            axes[row], (x0, x1), ("Gaussian", "Two moons")
        ):
            axis.scatter(points[:, 0], points[:, 1], s=4, alpha=0.35)
            axis.set_title(f"{split} · {name}")
            axis.set_aspect("equal")

    settings = {"moons_noise": MOONS_NOISE, "splits": SPLITS}
    (OUTPUT / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    figure.savefig(OUTPUT / "data_preview.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()

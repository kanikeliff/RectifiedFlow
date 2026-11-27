"""Step 18: results table, 2x4 sample grid, and trajectory comparison.
Run from the repository root (after eval_sw1.py and eval_straightness):
    python -m toy_2d.make_figures
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
FIGURES = RUNS / "figures"
STEPS = (1, 2, 5, 20)
MODELS = {"1rf": ("1-RF", "#257de0"), "2rf": ("2-RF", "#f47230")}  # validated pair
REFERENCE_GRAY = "#c8c8c8"
TEXT = "#191919"
NUM_TRAJECTORIES = 64
TRAJECTORY_SEED = 7          # which evaluation points are drawn as paths

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.edgecolor": "#999999",
    "axes.labelcolor": TEXT, "xtick.color": "#666666", "ytick.color": "#666666",
    "text.color": TEXT,
})


def shared_limits(arrays, margin=0.15):
    pts = np.concatenate(arrays)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = margin * (hi - lo).max()
    return (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)


def write_table(sw1, straight):
    rows = []
    for r in sw1["rows"]:
        for key, (label, _) in MODELS.items():
            rows.append({"model": label, "euler_steps": r["steps"],
                         "sw1": round(r[f"sw1_{key}"], 4)})
    with open(RUNS / "metrics" / "results_8rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "euler_steps", "sw1"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Euler steps | 1-RF SW1 ↓ | 2-RF SW1 ↓ | Absolute reduction | Ratio |",
             "|---:|---:|---:|---:|---:|"]
    for r in sw1["rows"]:
        a, b = r["sw1_1rf"], r["sw1_2rf"]
        lines.append(f"| {r['steps']} | {a:.4f} | {b:.4f} | {a - b:.4f} | {a / b:.2f}× |")
    lines += ["", f"Sampling floor (two independent moons samples): {sw1['floor']:.4f}  ",
              f"Raw Gaussian noise: {sw1['ceiling_gaussian']:.4f}", "",
              "| Model | Straightness S ↓ | Relative S |", "|---|---:|---:|"]
    for key, (label, _) in MODELS.items():
        res = straight["results"][key]
        lines.append(f"| {label} | {res['S']:.4f} | {res['S_relative']:.4f} |")
    text = "\n".join(lines) + "\n"
    (RUNS / "metrics" / "results_table.md").write_text(text, encoding="utf-8")
    print(text)


def sample_grid(outputs, sw1):
    ref = outputs["1rf"]["reference_x1"]
    arrays = [ref] + [outputs[k][f"euler_{s:02d}"] for k in MODELS for s in STEPS]
    xlim, ylim = shared_limits(arrays)
    sw_lookup = {(r["steps"], k): r[f"sw1_{k}"] for r in sw1["rows"] for k in MODELS}

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.0), sharex=True, sharey=True,
                             layout="constrained")
    for row, (key, (label, color)) in enumerate(MODELS.items()):
        for col, steps in enumerate(STEPS):
            ax = axes[row, col]
            pts = outputs[key][f"euler_{steps:02d}"]
            ax.scatter(ref[:, 0], ref[:, 1], s=2, c=REFERENCE_GRAY, linewidths=0,
                       label="Reference two moons")
            ax.scatter(pts[:, 0], pts[:, 1], s=2, c=color, linewidths=0, alpha=0.8,
                       label=f"{label} samples")
            ax.set_title(f"{label} · {steps} step{'s' if steps > 1 else ''}\n"
                         f"SW1 = {sw_lookup[(steps, key)]:.4f}")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.grid(color="#eeeeee", linewidth=0.6)
            ax.set_axisbelow(True)
    handles = [axes[0, 0].collections[0], axes[0, 0].collections[1], axes[1, 0].collections[1]]
    fig.legend(handles, [h.get_label() for h in handles], loc="outside lower center",
               ncol=3, frameon=False, markerscale=5)
    fig.savefig(FIGURES / "samples_2x4.png", dpi=200)
    plt.close(fig)


def trajectory_figure(trajectories, ref):
    rng = np.random.default_rng(TRAJECTORY_SEED)
    idx = rng.choice(trajectories["x0"].shape[0], NUM_TRAJECTORIES, replace=False)
    arrays = [ref] + [trajectories[k][:, idx].reshape(-1, 2) for k in MODELS]
    xlim, ylim = shared_limits(arrays, margin=0.05)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (key, (label, color)) in zip(axes, MODELS.items()):
        paths = trajectories[key][:, idx]                # (steps+1, K, 2)
        ax.scatter(ref[:, 0], ref[:, 1], s=2, c=REFERENCE_GRAY, linewidths=0,
                   label="Reference two moons")
        for k in range(paths.shape[1]):
            ax.plot(paths[:, k, 0], paths[:, k, 1], color=color, linewidth=1, alpha=0.7)
        ax.scatter(paths[0, :, 0], paths[0, :, 1], s=12, c="#666666", zorder=3,
                   label="Start (Gaussian)")
        ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=12, c=color, zorder=3,
                   edgecolors="white", linewidths=0.6, label=f"End ({label})")
        ax.set_title(f"{label}: {NUM_TRAJECTORIES} trajectories, same starting points")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=1.5)
    fig.savefig(FIGURES / "trajectories.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sw1 = json.loads((RUNS / "metrics" / "sw1.json").read_text())
    straight = json.loads((RUNS / "metrics" / "straightness.json").read_text())
    outputs = {k: np.load(RUNS / k / "eval_outputs.npz") for k in MODELS}
    trajectories = np.load(RUNS / "metrics" / "trajectories.npz")

    write_table(sw1, straight)
    sample_grid(outputs, sw1)
    trajectory_figure(trajectories, outputs["1rf"]["reference_x1"])
    print(f"saved figures to {FIGURES}")


if __name__ == "__main__":
    main()

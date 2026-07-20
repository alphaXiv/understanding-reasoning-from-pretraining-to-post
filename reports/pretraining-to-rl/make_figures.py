"""Render publication figures from the CSV values extracted from terminal logs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
SEED = pd.read_csv(HERE / "seed_summary.csv")
TRAJ = pd.read_csv(HERE / "trajectory_summary.csv")
COLORS = {"short": "#3B5CCC", "long": "#E07A32"}
LABELS = {"short": "8 shards · 9.18M tokens", "long": "32 shards · 36.70M tokens"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.25), constrained_layout=True)
for arm in ("short", "long"):
    frame = TRAJ[TRAJ.arm == arm]
    x = frame["update"].to_numpy()
    y = 100 * frame["mean_pass_at_1"].to_numpy()
    ci = 1.96 * 100 * frame["seed_standard_error"].to_numpy()
    axes[0].plot(x, y, marker="o", linewidth=2.5, color=COLORS[arm], label=LABELS[arm])
    axes[0].fill_between(x, y - ci, y + ci, color=COLORS[arm], alpha=0.14, linewidth=0)
axes[0].set_title("Held-out chess pass@1 during matched GRPO", loc="left")
axes[0].set_xlabel("GRPO update")
axes[0].set_ylabel("Pass@1 (%)")
axes[0].set_xticks([0, 5, 10, 20, 40, 60])
axes[0].grid(axis="y", alpha=0.2)
axes[0].legend(frameon=False, loc="upper left")
axes[0].text(60, 12.6, "+3.75 pp at update 60", ha="right", color=COLORS["long"], weight="bold")

short = SEED[SEED.arm == "short"].sort_values("seed")
long = SEED[SEED.arm == "long"].sort_values("seed")
for (_, s), (_, l) in zip(short.iterrows(), long.iterrows()):
    axes[1].plot(
        [0, 1],
        [100 * s.rl_slope_per_log10_update, 100 * l.rl_slope_per_log10_update],
        color="#AAB0B8",
        linewidth=1.3,
        marker="o",
        markersize=4,
        alpha=0.85,
    )
means = [100 * short.rl_slope_per_log10_update.mean(), 100 * long.rl_slope_per_log10_update.mean()]
axes[1].plot([0, 1], means, color="#171A21", linewidth=3.2, marker="o", markersize=7, zorder=5)
axes[1].set_title("Paired RL slope by seed", loc="left")
axes[1].set_ylabel("Pass@1 gain per log10 update (pp)")
axes[1].set_xticks([0, 1], ["8 shards", "32 shards"])
axes[1].set_xlim(-0.2, 1.2)
axes[1].grid(axis="y", alpha=0.2)
axes[1].text(
    0.5,
    max(means) + 0.65,
    "mean Δ +2.56 pp/decade\npaired 95% CI [+0.14, +4.99]",
    ha="center",
    va="bottom",
    weight="bold",
)
fig.suptitle("More pretraining produced faster RL improvement", fontsize=14, weight="bold")
fig.savefig(HERE / "images" / "primary_result.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
for (_, s), (_, l) in zip(short.iterrows(), long.iterrows()):
    ax.plot(
        [s.validation_loss, l.validation_loss],
        [100 * s.pass_at_1_update_60, 100 * l.pass_at_1_update_60],
        color="#B8BCC4",
        linewidth=1.1,
        zorder=1,
    )
for arm in ("short", "long"):
    frame = SEED[SEED.arm == arm]
    ax.scatter(
        frame.validation_loss,
        100 * frame.pass_at_1_update_60,
        color=COLORS[arm],
        s=46,
        label=LABELS[arm],
        zorder=3,
    )
ax.set_title("Lower pretraining loss ranked higher post-RL performance", loc="left")
ax.set_xlabel("Pretraining validation loss (lower is better)")
ax.set_ylabel("Pass@1 at GRPO update 60 (%)")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
ax.text(
    0.905,
    2.5,
    "Long arm: lower loss in 8/8 pairs\nand higher final pass@1 in 7/8 pairs",
    fontsize=9,
    weight="bold",
)
fig.savefig(HERE / "images" / "loss_ranking.png", dpi=180, bbox_inches="tight")
plt.close(fig)

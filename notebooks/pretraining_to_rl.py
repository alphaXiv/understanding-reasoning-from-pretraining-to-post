# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.14.17",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Does more chess pretraining make RL learn faster?

    This notebook is a self-contained walkthrough of completed Kubernetes evidence for
    *Understanding Reasoning from Pretraining to Post-Training* (arXiv:2607.16097).
    It embeds the measured results, so opening it does **not** rerun pretraining or GRPO.

    **Headline:** eight paired 20.54M-parameter models trained on 36.70M tokens had a
    steeper mean RL slope than models trained on 9.18M tokens: 0.05959 versus 0.03395
    pass@1 per log10 update. The paired difference was +0.02564 with a 95% interval
    from +0.00140 to +0.04989.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    updates = np.array([0, 5, 10, 20, 40, 60])
    short = np.array(
        [
            [0.03125, 0.03750, 0.04375, 0.04375, 0.06250, 0.08125],
            [0.03125, 0.03125, 0.05000, 0.04375, 0.05000, 0.06875],
            [0.04375, 0.04375, 0.03750, 0.03750, 0.03750, 0.06875],
            [0.01875, 0.02500, 0.02500, 0.03750, 0.07500, 0.07500],
            [0.08125, 0.08125, 0.08750, 0.07500, 0.11875, 0.12500],
            [0.01250, 0.01250, 0.01250, 0.01250, 0.01250, 0.03125],
            [0.09375, 0.09375, 0.09375, 0.10625, 0.12500, 0.15625],
            [0.00625, 0.01875, 0.01875, 0.05000, 0.03750, 0.05000],
        ]
    )
    long = np.array(
        [
            [0.05000, 0.06250, 0.06875, 0.08125, 0.08750, 0.10625],
            [0.06875, 0.06875, 0.07500, 0.09375, 0.11250, 0.12500],
            [0.06875, 0.05625, 0.06250, 0.08750, 0.12500, 0.14375],
            [0.01250, 0.01875, 0.02500, 0.03125, 0.03750, 0.05625],
            [0.08125, 0.08125, 0.07500, 0.12500, 0.14375, 0.16875],
            [0.03750, 0.04375, 0.04375, 0.03750, 0.07500, 0.06250],
            [0.10000, 0.10625, 0.10625, 0.13750, 0.17500, 0.18750],
            [0.03750, 0.03125, 0.03750, 0.03750, 0.09375, 0.10625],
        ]
    )
    seeds = np.arange(20260720, 20260728)
    return long, np, seeds, short, updates


@app.cell
def _(long, np, short, updates):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for values, label, color in (
        (short, "8 shards · 9.18M tokens", "#3B5CCC"),
        (long, "32 shards · 36.70M tokens", "#E07A32"),
    ):
        mean = 100 * values.mean(axis=0)
        ci = 1.96 * 100 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        ax.plot(updates, mean, marker="o", linewidth=2.5, label=label, color=color)
        ax.fill_between(updates, mean - ci, mean + ci, alpha=0.15, color=color)
    ax.set(title="Held-out pass@1 during matched GRPO", xlabel="GRPO update", ylabel="Pass@1 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig
    return


@app.cell
def _(mo, updates):
    position = mo.ui.slider(0, len(updates) - 1, value=len(updates) - 1, label="Inspect a matched update")
    position
    return (position,)


@app.cell
def _(long, mo, position, short, updates):
    index = position.value
    short_mean = 100 * short[:, index].mean()
    long_mean = 100 * long[:, index].mean()
    wins = int((long[:, index] > short[:, index]).sum())
    ties = int((long[:, index] == short[:, index]).sum())
    mo.md(
        f"""
        At **GRPO update {updates[index]}**, mean pass@1 is **{short_mean:.2f}%** for the
        short arm and **{long_mean:.2f}%** for the long arm, a difference of
        **{long_mean - short_mean:+.2f} percentage points**. The long arm wins
        **{wins}/8** paired seeds with **{ties}** tie(s).
        """
    )
    return


@app.cell
def _(long, mo, np, seeds, short):
    short_loss = np.array([1.04571962, 1.03059161, 1.03782117, 1.03676522, 1.02279603, 1.03188181, 1.01396275, 1.03248823])
    long_loss = np.array([0.87007725, 0.87347150, 0.87155890, 0.86512798, 0.87110144, 0.88014382, 0.86857718, 0.87519014])
    short_slope = np.array([0.03716121, 0.02613578, 0.01572979, 0.05430444, 0.04205928, 0.01225474, 0.05451730, 0.02939796])
    long_slope = np.array([0.03756779, 0.05389786, 0.08513499, 0.03062727, 0.08738074, 0.02531305, 0.08329582, 0.07349970])
    summary = mo.ui.table(
        {
            "seed": seeds,
            "short validation loss": short_loss,
            "long validation loss": long_loss,
            "short RL slope": short_slope,
            "long RL slope": long_slope,
            "short final pass@1": short[:, -1],
            "long final pass@1": long[:, -1],
        },
        selection=None,
    )
    summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A three-checkpoint robustness round

    A second completed Kubernetes round used 750 trace-SFT steps (three passes over
    500 available traces) and inserted a 16-shard checkpoint. Across eight seeds per
    exposure, the arm means were:

    | Tokens | Validation loss | RL slope | Pass@1 at update 60 |
    |---:|---:|---:|---:|
    | 9.18M | 1.02863 | 0.10178 | 21.33% |
    | 18.35M | 0.94129 | 0.10681 | 21.41% |
    | 36.70M | 0.87279 | 0.13810 | 27.19% |

    The mean order is consistent with the paper, but the short-to-midpoint endpoint
    is nearly flat and only 3/8 seeds have strictly increasing slopes across all three
    exposures. This is a useful warning: three ordered means do not establish a precise
    linear scaling law.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was held fixed?

    Both arms used the same 20,539,904-parameter architecture, seeds, public SFT traces,
    1,024-puzzle RL set, 160-puzzle held-out set, and GRPO hyperparameters. Only the nested
    pretraining prefix changed: 8 versus 32 official token shards. Every GPU trained a full
    independent seed; the two Kubernetes jobs allocated 16 NVIDIA RTX PRO 6000 Blackwell
    GPUs concurrently.

    ## What this does—and does not—show

    The longer-pretrained arm had lower validation loss in all eight pairs, higher mean
    pass@1 at every matched update, and a steeper mean RL slope. That reproduces the two
    selected directional claims at reduced scale. With only two exposure levels, it does
        not re-estimate the paper's approximately linear multi-checkpoint scaling law or its
        reported +0.84 slope–token correlation. The three-level robustness round above adds
        directional evidence while also showing substantial seed noise.
    """)
    return


if __name__ == "__main__":
    app.run()

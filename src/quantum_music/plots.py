"""Plotting functions shared by the whole team.

Rule: each function receives data that is already computed and only draws it.
It returns the matplotlib object and never calls ``plt.show()``, so it can be
used from notebooks, scripts and the app.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantum_music.circuits import NOTES
from quantum_music.theory import wilson_interval


def plot_counts(counts, theory, title="", ax=None):
    """Histogram of measured frequencies with Wilson error bars (95 %) and theoretical values.

    counts : {bitstring: number of occurrences}, for example {"00": 503, "11": 497}
    theory : {bitstring: theoretical probability}, for every possible outcome
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    keys = list(theory.keys())
    notes = NOTES[len(keys[0])]
    n = sum(counts.values())
    k = np.array([counts.get(key, 0) for key in keys])
    freq = k / n
    ci = np.array([wilson_interval(int(ki), n) for ki in k])
    yerr = np.clip(np.vstack([freq - ci[:, 0], ci[:, 1] - freq]), 0, None)
    x = np.arange(len(keys))
    ax.bar(x, freq, yerr=yerr, capsize=5, alpha=0.7,
           label=f"measured, {n} shots (Wilson 95 % CI)")
    ax.scatter(x, [theory[key] for key in keys], marker="_", s=900, color="black",
               linewidths=2, label="theory", zorder=3)
    # Values written above each bar: number of shots, measured %, theoretical %
    for xi, ki, fi, hi, key in zip(x, k, freq, ci[:, 1], keys):
        ax.text(xi, max(hi, theory[key]) + 0.03,
                f"{ki} / {n}\n{fi:.1%}  (theory {theory[key]:.0%})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{key}\n{notes[key]}" for key in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, markerscale=0.4)
    return ax


def save_fig(fig, path, dpi=150):
    """Save a figure (creating the folder if needed) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

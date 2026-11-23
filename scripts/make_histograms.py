"""Regenerate the Checkpoint 2 histograms (H and Bell circuits, no noise).

Run from the repository root:

    python scripts/make_histograms.py

Writes figures/histogram_hadamard.png and figures/histogram_bell.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from quantum_music.circuits import THEORY, build_bell, build_hadamard, run_counts
from quantum_music.plots import plot_counts, save_fig

OUT = Path(__file__).resolve().parents[1] / "figures"
SHOTS = 1000
SEED = 2026

CIRCUITS = {
    "hadamard": (build_hadamard(), "H circuit (1 qubit)"),
    "bell": (build_bell(), "Bell circuit (2 qubits)"),
}

for name, (circuit, title) in CIRCUITS.items():
    counts = run_counts(circuit, shots=SHOTS, seed=SEED)
    ax = plot_counts(counts, THEORY[name], title=f"{title}: {SHOTS} shots, no noise")
    path = save_fig(ax.figure, OUT / f"histogram_{name}.png")
    print(f"{name:<9} {counts}  ->  {path}")

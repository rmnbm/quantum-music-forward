"""Régénère les histogrammes du Checkpoint 2 (circuits H et Bell, sans bruit).

Lancer depuis la racine du dépôt :

    python scripts/make_histograms.py

Produit figures/histogram_hadamard.png et figures/histogram_bell.png.
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
    "hadamard": (build_hadamard(), "Circuit H (1 qubit)"),
    "bell": (build_bell(), "Circuit de Bell (2 qubits)"),
}

for name, (circuit, title) in CIRCUITS.items():
    counts = run_counts(circuit, shots=SHOTS, seed=SEED)
    ax = plot_counts(counts, THEORY[name], title=f"{title} : {SHOTS} shots, sans bruit")
    path = save_fig(ax.figure, OUT / f"histogram_{name}.png")
    print(f"{name:<9} {counts}  ->  {path}")

"""Fonctions de figures communes à toute l'équipe.

Règle : chaque fonction reçoit des données déjà calculées et ne fait que dessiner.
Elle renvoie l'objet matplotlib et n'appelle jamais ``plt.show()``, pour pouvoir
être utilisée dans les notebooks, les scripts et l'app.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantum_music.circuits import NOTES
from quantum_music.theory import wilson_interval


def plot_counts(counts, theory, title="", ax=None):
    """Histogramme des fréquences mesurées, barres d'erreur de Wilson (95 %) et valeur théorique.

    counts : {bitstring: nombre d'occurrences}, par exemple {"00": 503, "11": 497}
    theory : {bitstring: probabilité théorique}, toutes les issues possibles
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
           label=f"mesure, {n} shots (IC Wilson 95 %)")
    ax.scatter(x, [theory[key] for key in keys], marker="_", s=900, color="black",
               linewidths=2, label="théorie", zorder=3)
    # Valeurs écrites au-dessus de chaque barre : nombre de shots, % mesuré, % théorique
    for xi, ki, fi, hi, key in zip(x, k, freq, ci[:, 1], keys):
        ax.text(xi, max(hi, theory[key]) + 0.03,
                f"{ki} / {n}\n{fi:.1%}  (théorie {theory[key]:.0%})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{key}\n{notes[key]}" for key in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fréquence")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, markerscale=0.4)
    return ax


def save_fig(fig, path, dpi=150):
    """Enregistre une figure (crée le dossier si besoin) et la ferme."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

"""Tests des fonctions de figures. Lancer avec : pytest"""

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre graphique pendant les tests

from quantum_music.circuits import THEORY
from quantum_music.plots import plot_counts, save_fig


def test_hadamard_histogram_has_two_bars_and_notes():
    ax = plot_counts({"0": 512, "1": 488}, THEORY["hadamard"])
    assert len(ax.patches) == 2
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["0\nDo", "1\nRé"]


def test_bell_histogram_accepts_missing_outcomes():
    # Sans bruit, le simulateur ne renvoie pas les issues jamais observées (01, 10).
    ax = plot_counts({"00": 503, "11": 497}, THEORY["bell"])
    assert len(ax.patches) == 4
    assert ax.patches[1].get_height() == 0


def test_save_fig_writes_png(tmp_path):
    ax = plot_counts({"0": 500, "1": 500}, THEORY["hadamard"])
    out = save_fig(ax.figure, tmp_path / "sub" / "h.png")
    assert out.exists() and out.stat().st_size > 0

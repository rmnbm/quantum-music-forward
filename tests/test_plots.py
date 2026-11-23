"""Tests for the plotting functions. Run with: pytest"""

import matplotlib

matplotlib.use("Agg")  # no graphical window during tests

from quantum_music.circuits import THEORY
from quantum_music.plots import plot_counts, save_fig


def test_hadamard_histogram_has_two_bars_and_notes():
    ax = plot_counts({"0": 512, "1": 488}, THEORY["hadamard"])
    assert len(ax.patches) == 2
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["0\nDo", "1\nRé"]


def test_bell_histogram_accepts_missing_outcomes():
    # Without noise, the simulator does not return outcomes that never occur (01, 10).
    ax = plot_counts({"00": 503, "11": 497}, THEORY["bell"])
    assert len(ax.patches) == 4
    assert ax.patches[1].get_height() == 0


def test_save_fig_writes_png(tmp_path):
    ax = plot_counts({"0": 500, "1": 500}, THEORY["hadamard"])
    out = save_fig(ax.figure, tmp_path / "sub" / "h.png")
    assert out.exists() and out.stat().st_size > 0

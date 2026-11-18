import numpy as np
import matplotlib.pyplot as plt
from theory import wilson_interval

NOTES_1Q = {"0": "Do", "1": "Ré"}
NOTES_2Q = {"00": "Do", "01": "Ré", "10": "Mi", "11": "Fa"}
NOTES = {1: NOTES_1Q, 2: NOTES_2Q}

def plot_counts(counts, theory, title="", ax=None):
    """Frequency histogram with Wilson error bars and theoretical value."""    
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
    ax.bar(x, freq, yerr=yerr, capsize=5, alpha=0.7, label="mesure (IC Wilson 95 %)")
    ax.scatter(x, [theory[key] for key in keys], marker="_", s=900, color="black",
               linewidths=2, label="theory", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{key}\n{notes[key]}" for key in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax
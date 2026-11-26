"""Expériences de la partie bruit (Maïa).

Lancer depuis la racine du dépôt :

    python experiments/noise_experiments.py

Produit 5 figures dans figures/ et un résumé dans figures/noise_summary.txt :

1. noise_1_distributions.png   : Bell parfait vs Bell bruité (force 0,1), pour chaque bruit
2. noise_2_sweep.png           : écart à l'idéal quand on augmente la force du bruit
3. noise_3_depth.png           : état ou lecture ? le test de la profondeur
4. noise_4_mitigation.png      : état ou lecture ? la correction de lecture
5. noise_5_quantum_classical.png : à partir de quand le comportement devient classique
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantum_music.circuits import build_bell
from quantum_music.theory import tvd, wilson_interval
from quantum_music import noise as N

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

SHOTS = 4000
SEED = 2026
KEYS = ["00", "01", "10", "11"]
NOTES = {"00": "Do", "01": "Ré", "10": "Mi", "11": "Fa"}

# Une couleur et une forme de marqueur fixes par bruit (palette validée daltonisme)
COLORS = {"depolarizing": "#2a78d6", "amplitude_damping": "#eb6834",
          "phase_damping": "#1baf7a", "readout": "#eda100"}
MARKERS = {"depolarizing": "o", "amplitude_damping": "s",
           "phase_damping": "^", "readout": "D"}
IDEAL = "#9a9890"
INK, MUTED = "#2b2b29", "#6b6a63"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})

BELL = build_bell()
summary = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("figure ->", OUT / name)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)


# ---------------------------------------------------------------------------
# 1. Distributions : parfait vs bruité
# ---------------------------------------------------------------------------
P_DEMO = 0.1
ideal = N.exact_noisy_probabilities(BELL)
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
summary.append(f"1) Bell, force {P_DEMO}, {SHOTS} shots")
for ax, kind in zip(axes.flat, N.NOISE_KINDS):
    counts = N.run_noisy_counts(BELL, N.make_noise_model(**{kind: P_DEMO}), SHOTS, SEED)
    freq = np.array([counts[k] / SHOTS for k in KEYS])
    ci = np.array([wilson_interval(counts[k], SHOTS) for k in KEYS])
    yerr = np.clip(np.vstack([freq - ci[:, 0], ci[:, 1] - freq]), 0, None)
    x = np.arange(4)
    ax.bar(x - 0.2, [ideal[k] for k in KEYS], 0.38, color=IDEAL, label="parfait")
    ax.bar(x + 0.2, freq, 0.38, color=COLORS[kind], yerr=yerr, capsize=3,
           error_kw={"ecolor": INK, "lw": 1}, label="avec bruit")
    for xi, f in zip(x, freq):
        ax.text(xi + 0.2, f + 0.03, f"{f:.1%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{k}\n{NOTES[k]}" for k in KEYS])
    ax.set_ylim(0, 0.68)
    ax.set_title(N.LABELS[kind])
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    summary.append(f"   {N.LABELS[kind]:<32} " +
                   "  ".join(f"{k}={counts[k] / SHOTS:.3f}" for k in KEYS))
axes[0, 0].set_ylabel("fréquence")
axes[1, 0].set_ylabel("fréquence")
fig.suptitle(f"Circuit de Bell : parfait vs bruité (force {P_DEMO}, {SHOTS} mesures)",
             fontweight="bold")
save(fig, "noise_1_distributions.png")

# ---------------------------------------------------------------------------
# 2. Balayage de la force du bruit
# ---------------------------------------------------------------------------
fine = np.linspace(0, 0.5, 101)
coarse = np.linspace(0, 0.5, 11)
fig, ax = plt.subplots(figsize=(8, 4.5))
summary.append("\n2) Écart à l'idéal (TVD exacte) selon la force")
for kind in N.NOISE_KINDS:
    exact = [tvd(N.exact_noisy_probabilities(BELL, **{kind: p}), ideal) for p in fine]
    meas = [tvd(N.counts_to_probs(N.run_noisy_counts(
        BELL, N.make_noise_model(**{kind: p}), SHOTS, SEED + i)), ideal)
        for i, p in enumerate(coarse)]
    ax.plot(fine, exact, color=COLORS[kind], lw=2)
    ax.plot(coarse, meas, MARKERS[kind], color=COLORS[kind], ms=7,
            mec="white", mew=1.5, label=N.LABELS[kind])
    label_end(ax, fine[-1], exact[-1], N.LABELS[kind].split(" (")[0], COLORS[kind])
    summary.append(f"   {N.LABELS[kind]:<32} p=0.1 -> {exact[20]:.3f}   p=0.5 -> {exact[-1]:.3f}")
ax.set_xlabel("force du bruit")
ax.set_ylabel("écart à la musique parfaite (TVD)")
ax.set_xlim(0, 0.68)
ax.set_title("Plus le bruit est fort, plus on s'éloigne du cas parfait\n"
             "(lignes : calcul exact, points : simulation)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "noise_2_sweep.png")

# ---------------------------------------------------------------------------
# 3. Test de la profondeur : état ou lecture ?
# ---------------------------------------------------------------------------
depths = np.arange(0, 11)
P_DEPTH = {"depolarizing": 0.02, "amplitude_damping": 0.02,
           "phase_damping": 0.02, "readout": 0.05}
fig, ax = plt.subplots(figsize=(8, 4.5))
summary.append("\n3) P(01)+P(10) selon le nombre de paires CNOT ajoutées (0 -> 10)")
for kind in N.NOISE_KINDS:
    p = P_DEPTH[kind]
    exact = []
    for k in depths:
        pr = N.exact_noisy_probabilities(N.add_identity_layers(BELL, k), **{kind: p})
        exact.append(pr["01"] + pr["10"])
    meas = []
    for k in depths:
        c = N.run_noisy_counts(N.add_identity_layers(BELL, k),
                               N.make_noise_model(**{kind: p}), SHOTS, SEED + int(k))
        meas.append((c["01"] + c["10"]) / SHOTS)
    ax.plot(depths, exact, color=COLORS[kind], lw=2)
    ax.plot(depths, meas, MARKERS[kind], color=COLORS[kind], ms=7, mec="white",
            mew=1.5, label=f"{N.LABELS[kind]} (force {p})")
    label_end(ax, depths[-1], exact[-1], N.LABELS[kind].split(" (")[0], COLORS[kind])
    summary.append(f"   {N.LABELS[kind]:<32} {exact[0]:.3f} -> {exact[-1]:.3f}")
ax.set_xlabel("nombre de paires CNOT·CNOT ajoutées (ne changent rien en théorie)")
ax.set_ylabel("proportion de fausses notes (Ré + Mi)")
ax.set_xlim(-0.3, 13.5)
ax.set_title("Le bruit d'état grandit avec le nombre de portes,\n"
             "le bruit de lecture reste constant")
ax.legend(loc="upper left", fontsize=8)
save(fig, "noise_3_depth.png")

# ---------------------------------------------------------------------------
# 4. Correction de la lecture
# ---------------------------------------------------------------------------
scenarios = {
    "Lecture seule\n(lecture 0,05)": {"readout": 0.05},
    "Lecture + dépolarisant\n(lecture 0,05, dépol. 0,1)": {"readout": 0.05, "depolarizing": 0.1},
}
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
summary.append("\n4) Fausses notes P(01)+P(10) : brut -> après correction de lecture")
for ax, (name, st) in zip(axes, scenarios.items()):
    model = N.make_noise_model(**st)
    a = N.estimate_readout_matrix(model, shots=20000, seed=SEED)
    counts = N.run_noisy_counts(BELL, model, 20000, SEED + 1)
    raw = N.counts_to_probs(counts)
    fixed = N.mitigate_counts(counts, a)
    x = np.arange(4)
    ax.bar(x - 0.27, [ideal[k] for k in KEYS], 0.26, color=IDEAL, label="parfait")
    ax.bar(x, [raw[k] for k in KEYS], 0.26, color=COLORS["readout"], label="mesuré")
    ax.bar(x + 0.27, [fixed[k] for k in KEYS], 0.26, color=COLORS["depolarizing"],
           label="après correction")
    ax.set_xticks(x, [f"{k}\n{NOTES[k]}" for k in KEYS])
    ax.set_title(name)
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    ax.set_ylim(0, 0.62)
    r, f = raw["01"] + raw["10"], fixed["01"] + fixed["10"]
    ax.text(1.5, 0.3, f"fausses notes : {r:.1%} -> {f:.1%}", ha="center",
            color=INK, fontsize=9)
    summary.append(f"   {name.splitlines()[0]:<32} {r:.3f} -> {f:.3f}")
axes[0].set_ylabel("fréquence")
fig.suptitle("Corriger la lecture : ce qui reste après correction, c'est le bruit d'état",
             fontweight="bold")
save(fig, "noise_4_mitigation.png")

# ---------------------------------------------------------------------------
# 5. Quantique ou classique ?
# ---------------------------------------------------------------------------
grid = np.linspace(0, 1, 201)
pts = np.linspace(0, 1, 11)
fig, ax = plt.subplots(figsize=(8, 4.5))
summary.append("\n5) Fidélité de Bell F (F > 0,5 = intrication prouvée)")
for kind in N.NOISE_KINDS:
    # Au-delà de 0,5 une erreur de lecture n'a plus de sens (on lirait plus souvent
    # faux que juste) : on s'arrête à 0,5 pour elle.
    gmax = 0.5 if kind == "readout" else 1.0
    g = grid[grid <= gmax + 1e-9]
    pk = pts[pts <= gmax + 1e-9]
    F = np.array([N.bell_fidelity(N.bell_correlators(BELL, strengths={kind: p}))
                  for p in g])
    Fm = [N.bell_fidelity(N.bell_correlators(BELL, N.make_noise_model(**{kind: p}),
                                             SHOTS, SEED + 10 * i)) for i, p in enumerate(pk)]
    ax.plot(g, F, color=COLORS[kind], lw=2)
    ax.plot(pk, Fm, MARKERS[kind], color=COLORS[kind], ms=7, mec="white", mew=1.5,
            label=N.LABELS[kind])
    dy = {"phase_damping": 0.035, "amplitude_damping": -0.035}.get(kind, 0)
    label_end(ax, g[-1], F[-1] + dy, N.LABELS[kind].split(" (")[0], COLORS[kind])
    below = g[F <= 0.5 + 1e-9]
    seuil = f"{below[0]:.2f}" if len(below) else "jamais"
    verbe = ("PARAÎT classique (l'état est pourtant intact !)" if kind == "readout"
             else "devient classique")
    summary.append(f"   {N.LABELS[kind]:<32} F(0.1)={F[20]:.3f}  {verbe} à partir de : {seuil}")
ax.axhline(0.5, color=INK, lw=1, ls="--")
ax.annotate("lecture : fausse alarme,\nles qubits sont intacts", (0.21, 0.5),
            xytext=(0.5, 0.9), fontsize=8.5, color=INK,
            arrowprops={"arrowstyle": "->", "color": MUTED})
ax.text(0.01, 0.47, "limite classique (F = 0,5) : en dessous, des bits classiques suffisent",
        color=INK, fontsize=8.5, va="top")
ax.set_xlabel("force du bruit")
ax.set_ylabel("ressemblance avec Bell parfait (F)")
ax.set_xlim(0, 1.38)
ax.set_ylim(0, 1.05)
ax.set_title("À partir de quand les qubits se comportent comme des bits classiques ?\n"
             "(lignes : calcul exact, points : simulation)")
ax.legend(loc="upper right", fontsize=8)
save(fig, "noise_5_quantum_classical.png")

text = "\n".join(summary)
(OUT / "noise_summary.txt").write_text(text, encoding="utf-8")
print("\n" + text)

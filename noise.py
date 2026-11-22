"""Name 2 : modèles de bruit (Maïa).

Idée générale
-------------
Le circuit ne change pas : c'est la *machine* qui devient imparfaite.
On décrit ses défauts dans un ``NoiseModel`` de Qiskit Aer, puis on lance
le même circuit (celui de Morgane) sur ce simulateur bruité.

Quatre bruits sont étudiés :

- ``depolarizing``      : le qubit est remplacé par du hasard avec une proba p ;
- ``amplitude_damping`` : relaxation T1, un 1 retombe vers 0 (proba gamma) ;
- ``phase_damping``     : déphasage T2, le qubit perd son côté quantique (proba lambda) ;
- ``readout``           : l'état est bon, mais la mesure lit le mauvais bit (proba p).

Les trois premiers abîment l'ÉTAT. Ils sont appliqués après chaque porte H
(erreur sur 1 qubit) et chaque porte CNOT (erreur sur 2 qubits).
Le dernier abîme seulement la LECTURE : il est appliqué à la mesure.

Les rotations ``rx`` / ``ry`` qui servent à changer de base de mesure
ne reçoivent pas de bruit : ce sont des outils d'observation.

Convention Qiskit (little-endian) : dans "10", le qubit 0 est le bit de DROITE.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

NOISE_KINDS = ("depolarizing", "amplitude_damping", "phase_damping", "readout")
STATE_NOISE_KINDS = ("depolarizing", "amplitude_damping", "phase_damping")

# Portes qui reçoivent le bruit d'état
GATES_1Q = ["h"]
GATES_2Q = ["cx"]
# Portes que le simulateur garde telles quelles. Sans cette liste, Qiskit
# réécrirait X, RX, RY avec des portes H... qui recevraient alors du bruit.
BASIS_GATES = ["h", "cx", "x", "rx", "ry", "rz", "sx", "id"]

# Noms affichés dans les figures
LABELS = {
    "depolarizing": "Dépolarisant",
    "amplitude_damping": "Amortissement d'amplitude (T1)",
    "phase_damping": "Amortissement de phase (T2)",
    "readout": "Erreur de lecture",
}


# ---------------------------------------------------------------------------
# 1. Construire les erreurs et le modèle de bruit
# ---------------------------------------------------------------------------

def _check_strengths(strengths):
    for kind, value in strengths.items():
        if kind not in NOISE_KINDS:
            raise ValueError(f"bruit inconnu : {kind!r} (choix : {NOISE_KINDS})")
        values = value if isinstance(value, (tuple, list)) else (value,)
        for v in values:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"la force de {kind} doit être entre 0 et 1 (reçu {v})")


def state_errors(kind, p):
    """Erreurs (1 qubit, 2 qubits) pour un bruit d'état de force ``p``."""
    if kind == "depolarizing":
        return depolarizing_error(p, 1), depolarizing_error(p, 2)
    if kind == "amplitude_damping":
        e = amplitude_damping_error(p)
    elif kind == "phase_damping":
        e = phase_damping_error(p)
    else:
        raise ValueError(f"{kind!r} n'est pas un bruit d'état")
    # Sur une porte à 2 qubits, chaque qubit subit la même erreur
    return e, e.tensor(e)


def readout_matrix(p):
    """Matrice de lecture d'un qubit : ligne = vrai état, colonne = bit lu.

    ``p`` peut être un nombre (erreur symétrique) ou un couple
    ``(p01, p10)`` avec p01 = P(lire 1 | vrai 0) et p10 = P(lire 0 | vrai 1).
    """
    p01, p10 = (p, p) if np.isscalar(p) else p
    return np.array([[1 - p01, p01], [p10, 1 - p10]], dtype=float)


def _gate_errors(strengths):
    """Combine les bruits d'état demandés -> {"h": erreur, "cx": erreur}."""
    err1 = err2 = None
    for kind in STATE_NOISE_KINDS:
        p = strengths.get(kind, 0.0)
        if p > 0:
            e1, e2 = state_errors(kind, p)
            err1 = e1 if err1 is None else err1.compose(e1)
            err2 = e2 if err2 is None else err2.compose(e2)
    errors = {}
    if err1 is not None:
        errors.update({g: err1 for g in GATES_1Q})
        errors.update({g: err2 for g in GATES_2Q})
    return errors


def make_noise_model(**strengths):
    """Crée le modèle de bruit Qiskit Aer.

    Exemples ::

        make_noise_model(depolarizing=0.05)              # un seul bruit
        make_noise_model(phase_damping=0.1, readout=0.02)  # plusieurs bruits
        make_noise_model(readout=(0.01, 0.03))           # lecture asymétrique
    """
    _check_strengths(strengths)
    model = NoiseModel(basis_gates=BASIS_GATES)
    for gate, err in _gate_errors(strengths).items():
        model.add_all_qubit_quantum_error(err, [gate])
    ro = strengths.get("readout", 0.0)
    if np.any(np.asarray(ro) > 0):
        model.add_all_qubit_readout_error(ReadoutError(readout_matrix(ro)))
    return model


# ---------------------------------------------------------------------------
# 2. Exécuter un circuit sur la machine bruitée
# ---------------------------------------------------------------------------

def _simulator(noise_model):
    return AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()


def _transpile(circuit, sim):
    # optimization_level=0 : on interdit à Qiskit de simplifier le circuit
    # (sinon il supprimerait par exemple deux CNOT consécutifs, et leur bruit).
    return transpile(circuit, sim, optimization_level=0)


def run_noisy_counts(circuit, noise_model=None, shots=1000, seed=None):
    """Lance le circuit ``shots`` fois -> {bitstring: nombre d'occurrences}.

    Même rôle que ``run_counts`` de Morgane, mais avec un modèle de bruit.
    """
    sim = _simulator(noise_model)
    result = sim.run(_transpile(circuit, sim), shots=shots, seed_simulator=seed).result()
    counts = dict(result.get_counts())
    n = circuit.num_clbits
    return {format(i, f"0{n}b"): counts.get(format(i, f"0{n}b"), 0) for i in range(2 ** n)}


def run_noisy_samples(circuit, noise_model=None, shots=16, seed=None):
    """Résultats shot par shot, dans l'ordre : c'est la mélodie bruitée (pour Romain)."""
    sim = _simulator(noise_model)
    result = sim.run(_transpile(circuit, sim), shots=shots, seed_simulator=seed,
                     memory=True).result()
    return result.get_memory()


def counts_to_probs(counts):
    n = sum(counts.values())
    return {k: v / n for k, v in counts.items()}


# ---------------------------------------------------------------------------
# 3. Probabilités EXACTES avec bruit (sans hasard des shots)
# ---------------------------------------------------------------------------

def exact_noisy_state(circuit, **strengths):
    """Matrice densité exacte juste avant la mesure (bruit d'état seulement).

    Une matrice densité décrit aussi les états « abîmés », ce qu'un simple
    vecteur d'état ne peut pas faire.
    """
    _check_strengths(strengths)
    qc = circuit.remove_final_measurements(inplace=False)
    errors = _gate_errors(strengths)
    rho = DensityMatrix.from_label("0" * qc.num_qubits)
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        rho = rho.evolve(Operator(op), qargs)
        if op.name in errors:
            rho = rho.evolve(errors[op.name].to_quantumchannel(), qargs)
    return rho


def apply_readout(probs, p):
    """Effet de l'erreur de lecture sur une distribution {bitstring: proba}."""
    n = len(next(iter(probs)))
    keys = [format(i, f"0{n}b") for i in range(2 ** n)]
    vec = np.array([probs.get(k, 0.0) for k in keys])
    m1 = readout_matrix(p)
    m = m1
    for _ in range(n - 1):
        m = np.kron(m, m1)
    out = m.T @ vec
    return {k: float(v) for k, v in zip(keys, out)}


def exact_noisy_probabilities(circuit, **strengths):
    """Probabilités exactes de chaque résultat, bruit d'état ET de lecture compris.

    C'est la « théorie » avec bruit : on y compare les fréquences mesurées.
    Équivalent bruité de ``exact_probabilities`` de Morgane.
    """
    rho = exact_noisy_state(circuit, **{k: v for k, v in strengths.items() if k != "readout"})
    n = rho.num_qubits
    probs = {format(i, f"0{n}b"): float(p) for i, p in enumerate(rho.probabilities())}
    if np.any(np.asarray(strengths.get("readout", 0.0)) > 0):
        probs = apply_readout(probs, strengths["readout"])
    return probs


def bell_readout_formula(p):
    """Formule à la main : Bell parfait + erreur de lecture symétrique p."""
    same = ((1 - p) ** 2 + p ** 2) / 2
    diff = p * (1 - p)
    return {"00": same, "01": diff, "10": diff, "11": same}


def bell_depolarizing_formula(p):
    """Formule à la main : Bell avec dépolarisant p après H et après CNOT.

    Seul le dépolarisant du CNOT crée des 01 / 10 : P(01) = P(10) = p/4.
    """
    return {"00": 0.5 - p / 4, "01": p / 4, "10": p / 4, "11": 0.5 - p / 4}


# ---------------------------------------------------------------------------
# 4. Séparer bruit d'ÉTAT et bruit de LECTURE
# ---------------------------------------------------------------------------

def calibration_circuits(n=2):
    """Un circuit par résultat possible, SANS porte H ni CNOT.

    On prépare directement 00, 01, 10, 11 (avec des portes X, supposées
    parfaites) et on mesure. Comme il n'y a aucune porte bruitée,
    toute erreur observée vient forcément de la lecture.
    """
    circuits = {}
    for bits in ("".join(b) for b in product("01", repeat=n)):
        qc = QuantumCircuit(n, n, name=f"cal_{bits}")
        for q in range(n):
            if bits[n - 1 - q] == "1":  # little-endian
                qc.x(q)
        qc.measure(range(n), range(n))
        circuits[bits] = qc
    return circuits


def estimate_readout_matrix(noise_model, n=2, shots=4000, seed=None):
    """Mesure la matrice de lecture A : A[j, i] = P(lire j | préparé i)."""
    circuits = calibration_circuits(n)
    keys = list(circuits)
    a = np.zeros((2 ** n, 2 ** n))
    for i, bits in enumerate(keys):
        s = None if seed is None else seed + i
        counts = run_noisy_counts(circuits[bits], noise_model, shots, s)
        a[:, i] = [counts[k] / shots for k in keys]
    return a


def mitigate_counts(counts, a):
    """Corrige l'erreur de lecture : on « inverse » la matrice A.

    Ce qui reste d'écart à l'idéal après correction = bruit d'état.
    """
    keys = sorted(counts)
    q = np.array([counts[k] for k in keys], dtype=float)
    q /= q.sum()
    p, *_ = np.linalg.lstsq(a, q, rcond=None)
    p = np.clip(p, 0, None)
    p /= p.sum()
    return {k: float(v) for k, v in zip(keys, p)}


def add_identity_layers(circuit, k):
    """Ajoute k paires CNOT·CNOT avant la mesure.

    Une paire ne change rien en théorie (CNOT deux fois = rien),
    mais chaque CNOT ajoute du bruit d'état. Le bruit de lecture,
    lui, ne dépend pas du nombre de portes : c'est un test pour les séparer.
    (Circuit à 2 qubits, comme Bell.)
    """
    qc = circuit.remove_final_measurements(inplace=False)
    for _ in range(k):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return measure_in_basis(qc, "Z")


# ---------------------------------------------------------------------------
# 5. Quantique ou classique ? Corrélations dans plusieurs bases
# ---------------------------------------------------------------------------

def measure_in_basis(circuit, basis):
    """Ajoute une mesure dans la base demandée ("Z", "X" ou "Y") sur tous les qubits.

    Z : mesure normale. X et Y : on tourne d'abord le qubit, pour voir
    ce que la mesure normale ne voit pas (les phases, le côté quantique).
    """
    qc = circuit.remove_final_measurements(inplace=False)
    n = qc.num_qubits
    for q in range(n):
        if basis == "X":
            qc.ry(-np.pi / 2, q)
        elif basis == "Y":
            qc.rx(np.pi / 2, q)
        elif basis != "Z":
            raise ValueError("basis doit être 'Z', 'X' ou 'Y'")
    out = QuantumCircuit(n, n, name=f"{circuit.name}_{basis}")
    out.compose(qc, inplace=True)
    out.measure(range(n), range(n))
    return out


def correlator(probs):
    """<AB> = P(mêmes bits) - P(bits différents), entre -1 et +1.

    +1 : les deux voix sont toujours d'accord ; 0 : aucun lien ; -1 : toujours opposées.
    """
    total = sum(probs.values())
    same = probs.get("00", 0) + probs.get("11", 0)
    diff = probs.get("01", 0) + probs.get("10", 0)
    return (same - diff) / total


def bell_correlators(circuit, noise_model=None, shots=4000, seed=None, strengths=None):
    """Corrélations ZZ, XX, YY. Exactes si ``strengths`` est donné, sinon mesurées."""
    out = {}
    for i, b in enumerate("ZXY"):
        qc = measure_in_basis(circuit, b)
        if strengths is not None:
            out[b * 2] = correlator(exact_noisy_probabilities(qc, **strengths))
        else:
            s = None if seed is None else seed + i
            out[b * 2] = correlator(run_noisy_counts(qc, noise_model, shots, s))
    return out


def bell_fidelity(corr):
    """Ressemblance avec l'état de Bell parfait : F = (1 + XX - YY + ZZ) / 4.

    F = 1 : Bell parfait.  F > 1/2 : intrication prouvée (comportement quantique).
    F <= 1/2 : des bits classiques corrélés suffisent à expliquer les résultats.
    """
    return (1 + corr["XX"] - corr["YY"] + corr["ZZ"]) / 4


def classical_copy_correlators():
    """Modèle classique : on tire 00 ou 11 à pile ou face (bit copié).

    Il imite parfaitement Bell en base Z (ZZ = 1) mais pas en X ni Y.
    """
    return {"ZZ": 1.0, "XX": 0.0, "YY": 0.0}

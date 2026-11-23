import matplotlib.pyplot as plt

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

SIMULATOR = AerSimulator()

# Mapping bitstring -> note (slides du kickoff)
NOTES_1Q = {"0": "Do", "1": "Ré"}
NOTES_2Q = {"00": "Do", "01": "Ré", "10": "Mi", "11": "Fa"}
NOTES = {1: NOTES_1Q, 2: NOTES_2Q}

# Distributions théoriques, écrites à la main (pas lues du simulateur)
THEORY = {
    "hadamard": {"0": 0.5, "1": 0.5},
    "bell": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
}

def build_hadamard(measure=True):
    qc = QuantumCircuit(1, 1 if measure else 0, name="hadamard")
    qc.h(0)
    if measure:
        qc.measure(0, 0)
    return qc


def build_bell(measure=True):
    qc = QuantumCircuit(2, 2 if measure else 0, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    if measure:
        qc.measure([0, 1], [0, 1])
    return qc


def show_circuit(qc):
    """Dessine le circuit (matplotlib si possible, sinon texte)."""
    # Import local : IPython n'existe que dans Jupyter, pas dans l'app ni les tests.
    from IPython.display import display

    try:
        fig = qc.draw("mpl")
        display(fig)
        plt.close(fig)
    except Exception as e:
        print(f"(dessin matplotlib indisponible : {e}) -> version texte :")
        print(qc.draw("text"))


def exact_probabilities(circuit):
    """Probabilités exactes via le statevector"""
    qc = circuit.remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    n = qc.num_qubits
    keys = [format(i, f"0{n}b") for i in range(2 ** n)]
    return {k: float(probs.get(k, 0.0)) for k in keys}


def run_counts(circuit, shots=1000, seed=None):
    """Exécute le circuit `shots` fois -> dict {bitstring: nombre d'occurrences}."""
    tqc = transpile(circuit, SIMULATOR)
    result = SIMULATOR.run(tqc, shots=shots, seed_simulator=seed).result()
    return dict(result.get_counts())


def run_samples(circuit, shots=16, seed=None):
    """Exécute le circuit et renvoie les résultats shot par shot, dans l'ordre."""
    tqc = transpile(circuit, SIMULATOR)
    result = SIMULATOR.run(tqc, shots=shots, seed_simulator=seed, memory=True).result()
    return result.get_memory()
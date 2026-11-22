"""Tests de la partie bruit (Maïa). Lancer avec : pytest"""

import numpy as np
import pytest

from quantum_music.circuits import build_bell, build_hadamard
from quantum_music import noise as N

SHOTS = 20000


def within_4_sigma(freq, p, n=SHOTS):
    sigma = max(np.sqrt(p * (1 - p) / n), 1 / n)
    return abs(freq - p) <= 4 * sigma


# --- Sans bruit, on retrouve le cas parfait de Morgane ---------------------

def test_no_noise_gives_ideal_bell():
    probs = N.exact_noisy_probabilities(build_bell())
    assert probs == pytest.approx({"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5})


def test_zero_strength_model_has_no_forbidden_outcomes():
    counts = N.run_noisy_counts(build_bell(), N.make_noise_model(depolarizing=0.0),
                                shots=2000, seed=1)
    assert counts["01"] == 0 and counts["10"] == 0


def test_invalid_noise_rejected():
    with pytest.raises(ValueError):
        N.make_noise_model(bruit_inconnu=0.1)
    with pytest.raises(ValueError):
        N.make_noise_model(depolarizing=1.5)


# --- Chaque bruit donne l'effet attendu ------------------------------------

@pytest.mark.parametrize("p", [0.05, 0.2])
def test_depolarizing_matches_formula(p):
    exact = N.exact_noisy_probabilities(build_bell(), depolarizing=p)
    assert exact == pytest.approx(N.bell_depolarizing_formula(p))


@pytest.mark.parametrize("p", [0.02, 0.1])
def test_readout_matches_formula(p):
    exact = N.exact_noisy_probabilities(build_bell(), readout=p)
    assert exact == pytest.approx(N.bell_readout_formula(p))


def test_amplitude_damping_pushes_towards_00():
    exact = N.exact_noisy_probabilities(build_bell(), amplitude_damping=0.2)
    assert exact["00"] > 0.5 > exact["11"]
    assert exact["01"] == pytest.approx(exact["10"])


def test_phase_damping_invisible_in_z_but_visible_in_x():
    exact_z = N.exact_noisy_probabilities(build_bell(), phase_damping=0.3)
    assert exact_z == pytest.approx({"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5})
    corr = N.bell_correlators(build_bell(), strengths={"phase_damping": 0.3})
    assert corr["ZZ"] == pytest.approx(1.0)
    assert corr["XX"] < 0.9


def test_hadamard_readout():
    exact = N.exact_noisy_probabilities(build_hadamard(), readout=0.1)
    assert exact == pytest.approx({"0": 0.5, "1": 0.5})


@pytest.mark.parametrize("kind", N.NOISE_KINDS)
def test_simulation_agrees_with_exact(kind):
    p = 0.1
    exact = N.exact_noisy_probabilities(build_bell(), **{kind: p})
    counts = N.run_noisy_counts(build_bell(), N.make_noise_model(**{kind: p}),
                                shots=SHOTS, seed=11)
    for key, prob in exact.items():
        assert within_4_sigma(counts[key] / SHOTS, prob), (kind, key)


# --- Séparer état et lecture -------------------------------------------------

def test_calibration_sees_only_readout():
    model = N.make_noise_model(depolarizing=0.2, readout=0.05)
    a = N.estimate_readout_matrix(model, shots=SHOTS, seed=3)
    expected = np.kron(N.readout_matrix(0.05), N.readout_matrix(0.05)).T
    assert np.allclose(a, expected, atol=0.01)


def test_mitigation_removes_pure_readout_noise():
    model = N.make_noise_model(readout=0.05)
    a = N.estimate_readout_matrix(model, shots=SHOTS, seed=4)
    counts = N.run_noisy_counts(build_bell(), model, shots=SHOTS, seed=5)
    fixed = N.mitigate_counts(counts, a)
    assert fixed["01"] + fixed["10"] < 0.01


def test_depth_increases_state_noise_not_readout():
    bell = build_bell()
    dep = [N.exact_noisy_probabilities(N.add_identity_layers(bell, k), depolarizing=0.02)
           for k in (0, 5)]
    ro = [N.exact_noisy_probabilities(N.add_identity_layers(bell, k), readout=0.05)
          for k in (0, 5)]
    assert dep[1]["01"] > dep[0]["01"]
    assert ro[1] == pytest.approx(ro[0])


# --- Quantique ou classique ---------------------------------------------------

def test_ideal_bell_is_fully_quantum():
    corr = N.bell_correlators(build_bell(), strengths={})
    assert corr == pytest.approx({"ZZ": 1.0, "XX": 1.0, "YY": -1.0})
    assert N.bell_fidelity(corr) == pytest.approx(1.0)


def test_classical_copy_sits_at_threshold():
    assert N.bell_fidelity(N.classical_copy_correlators()) == pytest.approx(0.5)


def test_full_dephasing_becomes_classical():
    corr = N.bell_correlators(build_bell(), strengths={"phase_damping": 1.0})
    assert N.bell_fidelity(corr) == pytest.approx(0.5)


def test_noisy_samples_length():
    samples = N.run_noisy_samples(build_bell(), N.make_noise_model(readout=0.1),
                                  shots=16, seed=2)
    assert len(samples) == 16 and set(samples) <= {"00", "01", "10", "11"}

import numpy as np
import matplotlib.pyplot as plt


#THEORICAL VALUES
def std_error(p, n):
    """Écart-type de la fréquence observée : sqrt(p(1-p)/N)."""
    return float(np.sqrt(p * (1 - p) / n))


def wilson_interval(k, n, z=1.96):
    """Intervalle de confiance de Wilson (95 % par défaut) pour une proportion k/n.
    Contrairement à l'intervalle normal, il reste utile pour p proche de 0 ou 1."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    demi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def tvd(p, q):
    """Distance de variation totale entre deux distributions (dicts)."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
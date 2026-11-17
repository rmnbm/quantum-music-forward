# Quantum Music - Forward Problem


## Summary

Our team builds quantum circuits, studies gates, noise and measurements, and maps
measurement outcomes to musical notes. The guiding question is: why does the
music change when the circuit or the noise changes?

- 1 qubit: |0> -> H -> measure. Mapping: 0 -> C4 (Do), 1 -> D4 (Re).
- 2 qubits: Bell circuit (H on q0, CNOT q0 -> q1). Mapping: 00 -> C4, 01 -> D4, 10 -> E4, 11 -> F4.
- Noise models compared with the ideal circuit at equal shot count:
  depolarizing, amplitude damping (T1), phase damping (T2), readout bit-flip.
- Key question: does a given noise change the true state distribution, or only the measurement result?

Deliverables: Python code, reproducible figures, a scientific explanation and a
playable audio demo.

## Team

| Member   | Area                                  |
|----------|---------------------------------------|
| Morgane  | Circuits and theory                   |
| Maïa     | Noise models and experiments          |
| Romain   | Audio synthesis                       |
| Raphael  | Interactive app, plots, repository    |

## Installation

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the test suite:

```bash
pytest
```

Experiment scripts live in `experiments/` and write their outputs to `figures/`
and `audio/`. Run them from the repository root, for example:

```bash
python experiments/<script_name>.py
```

Generated figures and audio files are not tracked; rerun the scripts to
regenerate them.

## Repository layout

```
src/quantum_music/   Library code (circuits, theory, noise, audio, plots)
experiments/         Scripts that run experiments and produce figures and audio
app/                 Interactive demo for the Showroom
tests/               Unit tests (pytest)
notebooks/           Exploratory notebooks
figures/             Generated figures (not tracked)
audio/               Generated audio files (not tracked)
```

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

## Conventions

### Branches

- `main`: approved version only. Updated by merging `develop` at each checkpoint.
- `develop`: shared working branch. Nobody commits on it directly.
- Feature branches: one per task, created from `develop`, named
  `feature/<short-name>` (for example `feature/plots`, `feature/audio-mapping`).
  Delete them once merged.

### Workflow

1. `git switch develop && git pull`, then `git switch -c feature/<short-name>`.
2. Commit small steps with a clear message saying what changed
   (for example `Add readout noise model`), not `Add files via upload`.
3. Run `pytest` locally: all tests must pass.
4. Push the branch and open a pull request towards `develop`.
5. The review partner reads and approves it (Morgane <-> Maïa, Romain <-> Raphael),
   then the author merges.

Please do not upload files through the GitHub web interface: it skips the
tests and the review.

### Code

- Library code goes in `src/quantum_music/` and is imported as
  `from quantum_music.<module> import ...` (never `from theory import ...`).
- Library modules must not depend on Jupyter (`IPython`, `ipywidgets`): they are
  also used by the scripts, the tests and the app.
- Every new dependency goes into `requirements.txt`.
- Each module has its tests in `tests/test_<module>.py`.
- Plotting functions live in `plots.py`: they take already computed data,
  return the matplotlib object and never call `plt.show()`.
- Scripts and experiments fix their random seed so that figures are reproducible.

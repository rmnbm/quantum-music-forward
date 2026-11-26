"""Audio synthesis: turn measurement outcomes into sound.

Each bitstring is mapped to a note name by ``quantum_music.circuits.NOTES``,
then to a frequency by ``NOTE_FREQS``, and rendered as a numpy array of
samples in [-1, 1] at ``SAMPLE_RATE``. Functions return arrays and never
play sound, so they can be used from scripts, tests and the app.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from quantum_music.circuits import NOTES

SAMPLE_RATE = 44100


NOTE_FREQS = {"Do": 261.63, "Ré": 293.66, "Mi": 329.63, "Fa": 349.23}


def synth_note(freq, duration, sr=SAMPLE_RATE, amplitude=0.5):
    """Generate a pure sine tone.

    freq      : frequency in Hz (e.g. 261.63 for C4)
    duration  : length in seconds
    sr        : sample rate in samples per second
    amplitude : peak value, between 0 and 1
    Returns a 1-D numpy array of length int(sr * duration).
    """

    n = int(sr * duration)
    t = np.arange(n) / sr
    return apply_envelope(amplitude * np.sin(2 * np.pi * freq * t), sr=sr)


def apply_envelope(signal, sr=SAMPLE_RATE, attack=0.01, release=0.05):
    """Fade the signal in and out to avoid clicks.

    signal  : 1-D numpy array of samples
    sr      : sample rate in samples per second
    attack  : fade-in length in seconds
    release : fade-out length in seconds
    Returns a new array; the input is not modified.
    Ramps are shortened for notes shorter than attack + release.
    """

    n = len(signal)
    env = np.ones(n)
    a = min(int(sr * attack), int(n / 2))
    r = min(int(sr * release), int(n / 2))
    env[:a] *= np.linspace(0, 1, a)
    env[n - r :] *= np.linspace(1, 0, r)
    return signal * env


def samples_to_audio(samples, note_duration=0.25, sr=SAMPLE_RATE):
    """Turn a sequence of measurement outcomes into a melody.

    samples       : list of bitstrings in measurement order, as returned by
                    run_samples (e.g. ["11", "00", "00"])
    note_duration : length of each note in seconds
    sr            : sample rate in samples per second
    Each bitstring is mapped to a note name by NOTES, then to a frequency by
    NOTE_FREQS, and played for note_duration seconds; the notes are joined
    one after the other.
    Returns a 1-D numpy array of length len(samples) * int(sr * note_duration),
    or an empty array if samples is empty.
    """

    if len(samples) == 0:
        return np.zeros(0)

    notes = []
    for shot in samples:
        name = NOTES[len(shot)][shot]
        freq = NOTE_FREQS[name]
        notes.append(synth_note(freq, duration=note_duration, sr=sr))
    return np.concatenate(notes)


def save_wav(signal, path, sr=SAMPLE_RATE):
    """Write a signal to a 16-bit WAV file.

    signal : 1-D numpy array of samples in [-1, 1]; values outside are clipped
    path   : output file path (parent folders are created if needed)
    sr     : sample rate in samples per second
    Returns the path as a pathlib.Path.
    """

    clipped_signal = np.clip(signal, -1, 1)
    converted_signal = (clipped_signal * 32767).astype(np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, converted_signal)
    return path

"""Tests for the audio synthesis functions. Run with: pytest"""

import numpy as np
from scipy.io import wavfile

from quantum_music.audio import (
    NOTE_FREQS,
    apply_envelope,
    samples_to_audio,
    save_wav,
    synth_note,
)
from quantum_music.circuits import NOTES

SR = 8000  # small sample rate: fast tests, easy numbers


# --- synth_note ---


def test_synth_note_has_expected_length():
    assert len(synth_note(440, 0.5, sr=SR)) == 4000


def test_synth_note_stays_within_amplitude():
    x = synth_note(440, 0.5, sr=SR, amplitude=0.5)
    assert np.abs(x).max() <= 0.5


def test_synth_note_starts_and_ends_silent():
    # The envelope removes clicks: no jump at the start or the end of a note.
    x = synth_note(440, 0.5, sr=SR)
    assert x[0] == 0
    assert x[-1] == 0


def test_synth_note_has_requested_frequency():
    # One second of sound gives a frequency resolution of 1 Hz.
    x = synth_note(NOTE_FREQS["Mi"], 1.0, sr=SR)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    peak = freqs[np.argmax(spectrum)]
    assert abs(peak - NOTE_FREQS["Mi"]) <= 1


# --- apply_envelope ---


def test_envelope_handles_notes_shorter_than_attack_plus_release():
    x = apply_envelope(np.ones(10), sr=SR)
    assert len(x) == 10
    assert x[0] == 0 and x[-1] == 0


def test_envelope_does_not_modify_input():
    signal = np.ones(SR)
    apply_envelope(signal, sr=SR)
    assert signal.sum() == SR


# --- samples_to_audio ---


def test_melody_length_is_one_note_per_shot():
    x = samples_to_audio(["00", "11", "00"], note_duration=0.25, sr=SR)
    assert len(x) == 3 * 2000


def test_empty_samples_give_empty_signal():
    assert len(samples_to_audio([], sr=SR)) == 0


def test_melody_works_with_one_qubit():
    x = samples_to_audio(["0", "1"], note_duration=0.25, sr=SR)
    assert len(x) == 2 * 2000


def test_melody_plays_notes_in_measurement_order():
    x = samples_to_audio(["00", "11"], note_duration=0.25, sr=SR)
    do = synth_note(NOTE_FREQS["Do"], 0.25, sr=SR)
    fa = synth_note(NOTE_FREQS["Fa"], 0.25, sr=SR)
    assert np.allclose(x[:2000], do)
    assert np.allclose(x[2000:], fa)


def test_every_note_name_has_a_frequency():
    # Catches spelling mismatches with circuits.py (for example "Re" instead of "Ré").
    for mapping in NOTES.values():
        for name in mapping.values():
            assert name in NOTE_FREQS


# --- save_wav ---


def test_save_wav_writes_readable_16bit_file(tmp_path):
    x = synth_note(440, 0.5, sr=SR)
    out = save_wav(x, tmp_path / "sub" / "a.wav", sr=SR)
    sr, data = wavfile.read(out)
    assert sr == SR
    assert data.dtype == np.int16
    assert len(data) == len(x)


def test_save_wav_clips_instead_of_overflowing(tmp_path):
    # Without clipping, 2.0 * 32767 overflows int16 and becomes negative (a loud crack).
    out = save_wav(np.array([2.0, -2.0]), tmp_path / "c.wav", sr=SR)
    _, data = wavfile.read(out)
    assert list(data) == [32767, -32767]

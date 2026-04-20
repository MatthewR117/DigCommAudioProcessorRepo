from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


def linear_to_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Convert linear amplitude to decibels safely."""
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def envelope_follower(
    signal_abs: np.ndarray,
    sample_rate: int,
    attack_ms: float,
    release_ms: float
) -> np.ndarray:
    """
    Track the amplitude envelope using separate attack and release smoothing.
    """
    attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms * 0.001))
    release_coeff = np.exp(-1.0 / (sample_rate * release_ms * 0.001))

    env = np.zeros_like(signal_abs)
    prev = 0.0

    for i, sample in enumerate(signal_abs):
        if sample > prev:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        prev = coeff * prev + (1.0 - coeff) * sample
        env[i] = prev

    return env


def compress_channel(
    x: np.ndarray,
    sample_rate: int,
    threshold_db: float = -24.0,
    ratio: float = 8.0,
    attack_ms: float = 3.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 2.0
) -> np.ndarray:
    """
    Apply a simple feed-forward compressor to one audio channel.
    """
    x = x.astype(np.float64)

    env = envelope_follower(
        signal_abs=np.abs(x),
        sample_rate=sample_rate,
        attack_ms=attack_ms,
        release_ms=release_ms
    )

    env_db = linear_to_db(env)

    gain_reduction_db = np.zeros_like(env_db)
    over_threshold = env_db > threshold_db

    gain_reduction_db[over_threshold] = (
        threshold_db
        + (env_db[over_threshold] - threshold_db) / ratio
        - env_db[over_threshold]
    )

    total_gain_db = gain_reduction_db + makeup_gain_db
    total_gain_linear = db_to_linear(total_gain_db)

    y = x * total_gain_linear
    return y


def normalize_audio(x: np.ndarray, peak_target: float = 0.85) -> np.ndarray:
    """
    Normalize the signal so its absolute peak reaches peak_target.
    """
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x
    return x * (peak_target / peak)


def soft_limit_audio(x: np.ndarray, limit: float = 0.70) -> np.ndarray:
    """
    Soft limiter to reduce harsh peaks.
    """
    return limit * np.tanh(x / limit)


def process_audio(
    input_path: Path | str,
    output_path: Path | str,
    threshold_db: float = -24.0,
    ratio: float = 8.0,
    attack_ms: float = 3.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 2.0
) -> Path:
    """
    Read audio from input_path, compress it, limit it, normalize it,
    and write the processed audio to output_path.

    Returns the output path as a Path object.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load with librosa for broader file format support
    audio, sample_rate = librosa.load(str(input_path), sr=None, mono=False)

    # Convert to shape (samples, channels)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    else:
        audio = audio.T

    processed = np.zeros_like(audio, dtype=np.float64)

    for ch in range(audio.shape[1]):
        processed[:, ch] = compress_channel(
            x=audio[:, ch],
            sample_rate=sample_rate,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            makeup_gain_db=makeup_gain_db
        )

    processed = soft_limit_audio(processed, limit=0.70)
    processed = normalize_audio(processed, peak_target=0.85)

    sf.write(str(output_path), processed, sample_rate)

    return output_path

"""Acoustic feature vector for voice-emotion recognition (numpy/scipy only - no librosa/numba).

The SAME function is used for training (ml/train_voice_emotion.py) and inference, so features always match.
Per clip: MFCC (20) mean+std, delta-MFCC std, log-mel energy stats, spectral centroid/rolloff/flatness,
zero-crossing rate, RMS stats and Praat prosody (pitch, intensity, jitter, shimmer, HNR, voiced ratio).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
N_FFT, WIN, HOP = 512, 400, 160  # 25 ms window, 10 ms hop at 16 kHz
N_MELS, N_MFCC = 40, 20


def load_wav(path: str | Path, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        y = resample_poly(y, sr // g, file_sr // g).astype(np.float32)
    return y


def trim_silence(y: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    frames = _frames(y)
    if len(frames) == 0:
        return y
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return y
    return y[keep[0] * HOP: keep[-1] * HOP + WIN]


def _frames(y: np.ndarray) -> np.ndarray:
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    n = 1 + (len(y) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return y[idx]


@lru_cache
def _mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = 20.0,
                    fmax: float = 8000.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def spectral(y: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    from scipy.fft import dct

    y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # pre-emphasis
    frames = _frames(y) * np.hamming(WIN)
    mag = np.abs(np.fft.rfft(frames, n=N_FFT))
    power = mag ** 2 / N_FFT
    logmel = np.log(power @ _mel_filterbank(sr).T + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    psum = power.sum(axis=1) + 1e-12
    centroid = (power * freqs).sum(axis=1) / psum
    cum = np.cumsum(power, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flatness = np.exp(np.log(power + 1e-12).mean(axis=1)) / (power.mean(axis=1) + 1e-12)
    raw = _frames(y)
    zcr = (np.abs(np.diff(np.sign(raw), axis=1)) > 0).mean(axis=1)
    rms = np.sqrt((raw ** 2).mean(axis=1))
    return {"mfcc": mfcc, "logmel": logmel, "centroid": centroid, "rolloff": rolloff,
            "flatness": flatness, "zcr": zcr, "rms": rms}


def _prosody_vector(y: np.ndarray, sr: int = SR) -> list[float]:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) >= 5:
        st = 12 * np.log2(voiced / np.median(voiced))
        pitch = [float(np.log(np.median(voiced))), float(st.std()), float(np.ptp(np.percentile(st, [5, 95])))]
    else:
        pitch = [0.0, 0.0, 0.0]
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        jit = shim = 0.0
    hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
    vals = [jit, shim, hnr]
    vals = [0.0 if (v is None or np.isnan(v)) else float(v) for v in vals]
    return pitch + vals + [len(voiced) / max(len(f0), 1)]


def feature_vector(y: np.ndarray, sr: int = SR) -> np.ndarray:
    y = trim_silence(y)
    y = y / (np.abs(y).max() + 1e-9)  # loudness-normalise: emotion, not mic gain
    s = spectral(y, sr)
    mfcc = s["mfcc"]
    delta = np.diff(mfcc, axis=0) if len(mfcc) > 1 else np.zeros_like(mfcc)
    parts = [mfcc.mean(0), mfcc.std(0), delta.std(0),
             [s["logmel"].mean(), s["logmel"].std()]]
    for k in ("centroid", "rolloff", "flatness", "zcr", "rms"):
        parts.append([s[k].mean(), s[k].std()])
    parts.append(_prosody_vector(y, sr))
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])


FEATURE_DIM = N_MFCC * 3 + 2 + 5 * 2 + 7

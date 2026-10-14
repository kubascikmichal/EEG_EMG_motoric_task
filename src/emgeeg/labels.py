"""Moving / still labels from the EMG envelope (decision 1, step 2a).

The labelling envelope is deliberately slow (low-passed at ~2 Hz) and uses its
own thresholds, so a classifier working on 500 ms windows of EMG features can't
simply learn the labelling rule back. EMG-only results are still an upper bound.

Pipeline per run:
1. ``emg_envelope``: notch 50 Hz harmonics, band-pass, rectify, low-pass,
   normalize each channel to its own run percentile, average channels.
2. ``artifact_mask``: stretches far above the typical burst peak (cable or
   motion artifacts) and the build's ``bad`` samples become "artifact".
3. ``active_mask``: double threshold with hysteresis between the run's resting
   level and its typical burst level, then merge short gaps and drop short bursts.
4. ``to_segments``: contiguous (onset, offset, label) rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_closing
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt

STILL, MOVING, ARTIFACT = "still", "moving", "artifact"


def emg_envelope(x: np.ndarray, sf: float, band=(20.0, 240.0), lowpass_hz: float = 2.0,
                 norm_percentile: float = 95.0) -> np.ndarray:
    """Channel-averaged, per-channel-normalized envelope of ``x`` (n_channels, n)."""
    x = x - np.median(x, axis=1, keepdims=True)
    for f0 in np.arange(50.0, sf / 2 - 5, 50.0):
        b, a = iirnotch(f0, 30.0, sf)
        x = filtfilt(b, a, x, axis=1)
    x = sosfiltfilt(butter(4, band, "bandpass", fs=sf, output="sos"), x, axis=1)
    env = sosfiltfilt(butter(2, lowpass_hz, "low", fs=sf, output="sos"), np.abs(x), axis=1)
    env /= np.percentile(env, norm_percentile, axis=1, keepdims=True)
    return env.mean(axis=0)


def _closing(mask: np.ndarray, n: int) -> np.ndarray:
    """Fill False gaps shorter than ``n`` samples (edges untouched)."""
    if n < 1 or not mask.any():
        return mask
    pad = np.pad(mask, n)
    return binary_closing(pad, structure=np.ones(n + 1, dtype=bool))[n:-n]


def _drop_short(mask: np.ndarray, n: int) -> np.ndarray:
    """Remove True runs shorter than ``n`` samples."""
    out = mask.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < n:
            out[s:e] = False
    return out


def levels(env: np.ndarray, sf: float, exclude: np.ndarray | None = None,
           rest_percentile: float = 10.0, min_rise: float = 0.5) -> tuple[float, float]:
    """(resting level, typical burst level) of a run's envelope.

    The burst level is the median height of the envelope peaks (at least 1 s
    apart), so a minority of artifact peaks can't shift it. Peaks must rise by
    at least ``min_rise`` × the resting level, so in a run that is mostly rest
    the noise wiggles don't count as bursts.
    """
    e = env if exclude is None or not exclude.any() else np.where(exclude, np.nan, env)
    rest = np.nanpercentile(e, rest_percentile)
    spread = np.nanpercentile(e, 90) - rest
    prominence = max(0.2 * spread, min_rise * rest)
    peaks, _ = find_peaks(np.nan_to_num(e, nan=rest), distance=int(sf), prominence=prominence)
    burst = np.median(e[peaks]) if len(peaks) >= 5 else np.nanpercentile(e, 99)
    return float(rest), float(burst)


def artifact_mask(env: np.ndarray, sf: float, bad: np.ndarray | None = None, factor: float = 3.0,
                  pad_s: float = 0.5, merge_s: float = 2.0) -> np.ndarray:
    """True where the envelope exceeds ``factor`` × the typical burst level, or ``bad`` is set."""
    rest, burst = levels(env, sf)
    art = env > rest + factor * (burst - rest)
    if bad is not None:
        art |= bad.astype(bool)
    art = binary_dilation(art, structure=np.ones(2 * int(pad_s * sf) + 1, dtype=bool)) if art.any() else art
    return _closing(art, int(merge_s * sf))


def active_mask(env: np.ndarray, sf: float, exclude: np.ndarray | None = None,
                high: float = 0.35, low: float = 0.15,
                min_gap_s: float = 0.2, min_burst_s: float = 0.2) -> np.ndarray:
    """Bursts: start above ``high`` and extend while above ``low`` (fractions of rest→burst level).

    ``exclude`` (e.g. artifacts) is left out when estimating the levels.
    """
    rest, burst = levels(env, sf, exclude)
    above_low = env > rest + low * (burst - rest)
    above_high = env > rest + high * (burst - rest)
    edges = np.flatnonzero(np.diff(np.r_[0, above_low.astype(np.int8), 0]))
    act = np.zeros_like(above_low)
    for s, e in zip(edges[::2], edges[1::2]):
        if above_high[s:e].any():
            act[s:e] = True
    act = _closing(act, int(min_gap_s * sf))
    return _drop_short(act, int(min_burst_s * sf))


def to_segments(time: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Contiguous runs of equal labels as (onset_s, offset_s, label); offset = next onset."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    dt = np.median(np.diff(time))
    return pd.DataFrame({
        "onset_s": time[starts],
        "offset_s": np.r_[time[change], time[-1] + dt],
        "label": labels[starts],
    }).round({"onset_s": 4, "offset_s": 4})


def label_run(x: np.ndarray, time: np.ndarray, sf: float, bad: np.ndarray | None = None,
              envelope_kw: dict | None = None, artifact_kw: dict | None = None,
              active_kw: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(segments, per-sample labels, envelope) for one run's EMG."""
    env = emg_envelope(x, sf, **(envelope_kw or {}))
    art = artifact_mask(env, sf, bad, **(artifact_kw or {}))
    act = active_mask(env, sf, art, **(active_kw or {}))
    lab = np.where(art, ARTIFACT, np.where(act, MOVING, STILL)).astype(object)
    return to_segments(time, lab), lab, env

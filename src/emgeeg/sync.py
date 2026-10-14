"""Removing filler samples and reconstructing per-sample timestamps.

Why not the stored ``time`` arrays: in some runs the EMG ``time`` array is up to
±0.7 s away from the chunk arrival clock, and where filler blocks were inserted
it is compressed (1.95 ms steps), which produced the apparent 0.6–1.36 s
EEG–EMG start offsets. The chunk arrival clock (``local_clock``) is shared by
both devices and is not affected.

Filler: during dropouts the EMG driver inserts blocks of 100 zero samples with
``Streaming == 0``. The signal is continuous across them (the step over a block
is the size of a normal sample-to-sample step), so no time passed: they are
removed, not masked.

Timing model: chunk k arrives at ``local_clock[k] = t(last sample of k) + latency``.
A straight-line fit of arrival vs. sample count gives the effective sampling
rate; a running median of the residuals (``smooth_s``) follows slow clock
wander; a low quantile of what is left estimates the minimum transfer latency,
which is subtracted. Both devices are assumed to have the same minimum latency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .io import DeviceRecording


@dataclass
class Timing:
    t: np.ndarray            # timestamp of every kept sample, on the shared host clock (s)
    sf_effective: float      # fitted sampling rate
    residual_sd_ms: float    # arrival jitter left after the fit
    wander_ms: float         # peak-to-peak of the slow correction


def filler_mask(rec: DeviceRecording) -> np.ndarray:
    """True for samples the driver inserted during dropouts (``Streaming == 0``)."""
    return rec.channel("Streaming") == 0


def fit_timing(chunk_real: np.ndarray, chunk_clock: np.ndarray, sf_nominal: float,
               smooth_s: float = 20.0, latency_quantile: float = 0.05) -> Timing:
    """Timestamps for ``chunk_real.sum()`` samples from chunk arrival times.

    ``chunk_real`` is the number of real (non-filler) samples in each chunk.
    """
    keep = chunk_real > 0
    n_real, clock = chunk_real[keep], chunk_clock[keep]
    last_idx = np.cumsum(n_real) - 1          # index of the last sample in each chunk

    slope, intercept = np.polyfit(last_idx, clock, 1)
    resid = clock - (intercept + slope * last_idx)
    win = max(3, int(round(smooth_s * sf_nominal / np.median(n_real))) | 1)
    wander = median_filter(resid, size=win, mode="nearest")
    latency = np.quantile(resid - wander, latency_quantile)

    idx = np.arange(int(n_real.sum()))
    t = intercept + slope * idx + np.interp(idx, last_idx, wander) + latency
    return Timing(
        t=t,
        sf_effective=1.0 / slope,
        residual_sd_ms=float(np.std(resid - wander) * 1e3),
        wander_ms=float(np.ptp(wander) * 1e3),
    )


def device_timing(rec: DeviceRecording, **kw) -> tuple[np.ndarray, Timing]:
    """(mask of samples to keep, timing of the kept samples) for one device."""
    keep = ~filler_mask(rec)
    bounds = np.cumsum(rec.chunk_sizes)[:-1]
    chunk_real = np.array([k.sum() for k in np.split(keep, bounds)])
    return keep, fit_timing(chunk_real, rec.chunk_clock, rec.sf, **kw)

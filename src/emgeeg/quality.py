"""Per-sample quality mask."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def saturation_mask(x: np.ndarray, limit: float | None, sf: float, margin_s: float) -> np.ndarray:
    """True where any channel of ``x`` (n_channels, n) is at |value| >= ``limit``, widened by ``margin_s``."""
    if limit is None:
        return np.zeros(x.shape[1], dtype=bool)
    bad = (np.abs(x) >= float(limit)).any(axis=0)
    margin = int(round(margin_s * sf))
    if margin and bad.any():
        bad = binary_dilation(bad, structure=np.ones(2 * margin + 1, dtype=bool))
    return bad

from pathlib import Path

import numpy as np

from emgeeg.anonymize import RunId, assign_run_ids
from emgeeg.quality import saturation_mask
from emgeeg.sync import fit_timing


def test_runs_numbered_by_time_within_session():
    names = [
        "subj-1_ses-S002_task-Fusion_run-001_20000101_120500_eeg.db",
        "subj-1_ses-S002_task-Fusion_run-001_20000101_110000_eeg.db",
        "subj-1_ses-S001_task-Fusion_run-002_20000101_113000_eeg.db",
        "subj-1_ses-S001_task-Fusion_run-001_20000101_100000_eeg.db",
    ]
    ids = assign_run_ids([Path(n) for n in names])
    assert ids[Path(names[1])] == RunId(1, 2, "fusion", 1)
    assert ids[Path(names[0])] == RunId(1, 2, "fusion", 2)
    assert ids[Path(names[3])].stem == "sub-01_ses-01_task-fusion_run-01"


def test_fit_timing_recovers_rate_and_start():
    rng = np.random.default_rng(0)
    sf_true, t_start, n_chunks = 499.85, 1000.0, 4000
    sizes = np.full(n_chunks, 25)
    last = np.cumsum(sizes) - 1
    latency = 0.010 + rng.exponential(0.020, n_chunks)
    clock = t_start + last / sf_true + latency
    tm = fit_timing(sizes, clock, sf_nominal=500.0)
    assert abs(tm.sf_effective - sf_true) < 0.01
    assert len(tm.t) == sizes.sum()
    # Start is recovered up to the (unknown, shared) minimum latency.
    assert abs(tm.t[0] - (t_start + 0.010)) < 0.005


def test_saturation_mask_margin():
    x = np.zeros((2, 100))
    x[1, 50] = -2.25e6
    bad = saturation_mask(x, 2.2e6, sf=100.0, margin_s=0.02)
    assert bad.sum() == 5 and bad[48:53].all()
    assert not saturation_mask(x, None, 100.0, 0.02).any()

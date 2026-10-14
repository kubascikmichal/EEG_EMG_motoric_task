"""Mapping original recording names to anonymized ones.

Original: ``subj-1_ses-S002_task-Fusion_run-001_YYYYMMDD_HHMMSS_eeg.db``
Output:   ``sub-01_ses-01_task-fusion_run-01_{EEG,EMG}``

Runs are renumbered by time order within each subject + session (decision 3),
which also resolves the duplicate run-001 in sub-1 ses-2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

_NAME = re.compile(
    r"subj-(?P<sub>\d+)_ses-S(?P<ses>\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)"
    r"_(?P<date>\d{8})_(?P<time>\d{6})_eeg\.db$")


@dataclass(frozen=True)
class RunId:
    sub: int
    ses: int
    task: str
    run: int

    @property
    def stem(self) -> str:
        return f"sub-{self.sub:02d}_ses-{self.ses:02d}_task-{self.task}_run-{self.run:02d}"

    @property
    def session_dir(self) -> str:
        return f"ses-{self.ses:02d}"


def assign_run_ids(paths: list[Path]) -> dict[Path, RunId]:
    """Anonymized id for every .db path; runs numbered by start time per subject + session."""
    parsed = []
    for p in paths:
        m = _NAME.search(p.name)
        if not m:
            raise ValueError(f"unexpected recording name: {p.name}")
        parsed.append((int(m["sub"]), int(m["ses"]), m["task"].lower(), m["date"] + m["time"], p))
    parsed.sort()
    out: dict[Path, RunId] = {}
    for (sub, ses, task), group in groupby(parsed, key=lambda x: x[:3]):
        for run, (*_, p) in enumerate(group, start=1):
            out[p] = RunId(sub, ses, task, run)
    return out

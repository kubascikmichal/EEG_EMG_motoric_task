"""Reading the recorder's SQLite (.db) files.

Each .db holds both devices. Per device there are three tables keyed by a UUID:
``meta_<uuid>`` (one row: channel names, types, units, sf, device id),
``data_<uuid>`` (one row per received chunk: ``data`` and ``time`` are
``np.save`` blobs of shape (n_channels, n) and (n,), ``local_clock`` is the
host arrival time of the chunk) and ``annotations_<uuid>`` (empty).
"""
from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Device model in the meta id -> modality.
DEVICE_MODALITY = {"MAXI": "EEG", "MINI": "EMG"}


@dataclass
class DeviceRecording:
    modality: str            # "EEG" or "EMG"
    channels: list[str]      # all channel names, in data row order
    units: list[str]
    sf: float                # nominal sampling rate
    data: np.ndarray         # (n_channels, n_samples), as stored (signals in µV)
    chunk_sizes: np.ndarray  # samples per received chunk
    chunk_clock: np.ndarray  # host arrival time of each chunk (s)

    def channel(self, name: str) -> np.ndarray:
        return self.data[self.channels.index(name)]

    @property
    def signal_channels(self) -> list[str]:
        prefix = "EMG" if self.modality == "EMG" else ""
        n = 8 if self.modality == "EMG" else 32
        names = self.channels[:n]
        assert all(c.startswith(prefix) and not c.startswith("contact_") for c in names), names
        return names


def _load_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def read_db(path: str | Path) -> dict[str, DeviceRecording]:
    """Read both devices from one .db file, keyed by modality."""
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)  # immutable: no -wal/-shm files
    try:
        out: dict[str, DeviceRecording] = {}
        metas = [n for (n,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'meta_%'")]
        for meta in metas:
            uid = meta[len("meta_"):]
            channels, _types, units, sf, dev_id = con.execute(f"SELECT * FROM `{meta}`").fetchone()
            model = next(m for m in DEVICE_MODALITY if f"_{m}_" in dev_id)
            rows = con.execute(
                f"SELECT data, local_clock FROM `data_{uid}` ORDER BY rowid").fetchall()
            chunks = [_load_blob(d) for d, _ in rows]
            rec = DeviceRecording(
                modality=DEVICE_MODALITY[model],
                channels=channels.split(","),
                units=units.split(","),
                sf=float(sf),
                data=np.concatenate(chunks, axis=1),
                chunk_sizes=np.array([c.shape[1] for c in chunks]),
                chunk_clock=np.array([lc for _, lc in rows], dtype=float),
            )
            out[rec.modality] = rec
        return out
    finally:
        con.close()

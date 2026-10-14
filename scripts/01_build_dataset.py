"""Step 1: build the anonymized, synchronized dataset from the raw .db files.

Output per run and modality (see docs/PLAN.md, decisions 3–5):
    dataset/processed/ses-XX/sub-XX_ses-XX_task-fusion_run-XX_{EEG,EMG}.csv
        time (s from run start, shared by both devices), signal channels (µV), bad (0/1)
    ... same stem .json sidecar
and dataset/processed/runs.csv, one row per run and modality.

Usage: python scripts/01_build_dataset.py [--config config.yaml] [--check-csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emgeeg.anonymize import assign_run_ids  # noqa: E402
from emgeeg.io import read_db  # noqa: E402
from emgeeg.quality import saturation_mask  # noqa: E402
from emgeeg.sync import device_timing  # noqa: E402

DEVICE_MODEL = {"EEG": "MAXI", "EMG": "MINI"}


def check_against_csv(db_path: Path, rec, csv_dir: Path) -> str:
    """Confirm the original CSV export holds the same signal values (CSV is in V, .db in µV)."""
    pattern = f"{db_path.stem}_*_{DEVICE_MODEL[rec.modality]}_*-raw.csv"
    found = list(csv_dir.glob(pattern))
    if len(found) != 1:
        return f"csv not found ({len(found)} matches)"
    chans = rec.signal_channels
    csv = pd.read_csv(found[0], usecols=chans)[chans].to_numpy().T * 1e6
    db = rec.data[: len(chans)]
    # The export drops leading samples to start both devices together; find that offset.
    first = csv[0, :50]
    offsets = [o for o in range(db.shape[1] - csv.shape[1] + 1)
               if np.allclose(db[0, o:o + 50], first, rtol=1e-6, atol=1e-3)]
    if not offsets:
        return "csv does not match"
    o = offsets[0]
    ok = np.allclose(db[:, o:o + csv.shape[1]], csv, rtol=1e-6, atol=1e-3)
    return f"{'match' if ok else 'MISMATCH'} (csv starts {o} samples later, ends {db.shape[1] - o - csv.shape[1]} earlier)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--check-csv", action="store_true", help="cross-check values against csv/original")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    raw = ROOT / cfg["paths"]["raw"]
    out_root = ROOT / cfg["paths"]["processed"]
    q = cfg["build"]["quality"]

    run_ids = assign_run_ids(sorted(raw.glob("sub-*/ses-*/*.db")))
    rows = []
    for db_path, rid in run_ids.items():
        recs = read_db(db_path)
        timed = {m: device_timing(r, **cfg["build"]["timing"]) for m, r in recs.items()}
        t0 = min(tm.t[0] for _, tm in timed.values())
        out_dir = out_root / rid.session_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for modality, rec in sorted(recs.items()):
            keep, tm = timed[modality]
            chans = rec.signal_channels
            x = rec.data[: len(chans)][:, keep]
            bad = saturation_mask(x, q["saturation_uv"][modality], rec.sf, q["margin_s"])
            time = tm.t - t0
            stem = f"{rid.stem}_{modality}"

            df = pd.DataFrame(x.T, columns=chans)
            df.insert(0, "time", time)
            df["bad"] = bad.astype(np.int8)
            df.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")

            side = {
                "subject": rid.sub, "session": rid.ses, "task": rid.task, "run": rid.run,
                "modality": modality,
                "channels": chans,
                "units": "microvolts",
                "sf_nominal": rec.sf,
                "sf_effective": round(tm.sf_effective, 4),
                "n_samples": int(x.shape[1]),
                "duration_s": round(float(time[-1] - time[0]), 3),
                "start_offset_s": round(float(time[0]), 4),
                "time_reference": "shared host clock; 0 = first sample of whichever device started first",
                "filler_samples_removed": int((~keep).sum()),
                "bad_samples": int(bad.sum()),
                "timing_residual_sd_ms": round(tm.residual_sd_ms, 1),
                "timing_wander_ms": round(tm.wander_ms, 1),
            }
            if args.check_csv:
                side["csv_check"] = check_against_csv(db_path, rec, ROOT / cfg["paths"]["csv_original"])
            (out_dir / f"{stem}.json").write_text(json.dumps(side, indent=2) + "\n")
            rows.append({k: v for k, v in side.items() if k not in ("channels", "units", "time_reference")})
            print(f"{stem}: n={side['n_samples']} start={side['start_offset_s']:+.3f}s "
                  f"fs={side['sf_effective']} filler={side['filler_samples_removed']} bad={side['bad_samples']}"
                  + (f" csv={side['csv_check']}" if args.check_csv else ""))

    pd.DataFrame(rows).to_csv(out_root / "runs.csv", index=False)


if __name__ == "__main__":
    main()

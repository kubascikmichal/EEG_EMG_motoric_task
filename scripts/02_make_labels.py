"""Step 2a: moving / still / artifact labels from the EMG envelope.

Reads dataset/processed/ses-XX/*_EMG.csv (step 1) and writes, per run:
    dataset/processed/ses-XX/sub-XX_ses-XX_task-fusion_run-XX_labels.csv
        onset_s, offset_s, label   (times on the run's shared clock)
    dataset/processed/review/sub-XX_..._run-XX.png   (for checking by eye)
and dataset/processed/labels_summary.csv.

Usage: python scripts/02_make_labels.py [--config config.yaml] [--no-plots]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emgeeg.labels import ARTIFACT, MOVING, label_run, levels  # noqa: E402

COLORS = {MOVING: "tab:orange", ARTIFACT: "tab:red"}


def review_plot(path: Path, time, env, segments, rest, burst, cfg, title, row_s=60.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = int(np.ceil((time[-1] - time[0]) / row_s))
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 1.6 * n_rows), squeeze=False)
    step = max(1, len(time) // 200_000)
    hi = rest + cfg["active"]["high"] * (burst - rest)
    lo = rest + cfg["active"]["low"] * (burst - rest)
    ymax = max(1.5 * burst, 1e-9)
    for r, ax in enumerate(axes[:, 0]):
        a = time[0] + r * row_s
        for seg in segments.itertuples():
            if seg.label in COLORS and seg.offset_s > a and seg.onset_s < a + row_s:
                ax.axvspan(seg.onset_s, seg.offset_s, color=COLORS[seg.label], alpha=0.25, lw=0)
        ax.plot(time[::step], env[::step], "k", lw=0.6)
        for y, ls in ((hi, "--"), (lo, ":")):
            ax.axhline(y, color="tab:blue", ls=ls, lw=0.7)
        ax.set_xlim(a, a + row_s)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_title(f"{title}   orange = moving, red = artifact, blue = thresholds", loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    cfg_all = yaml.safe_load(args.config.read_text())
    cfg = cfg_all["labels"]
    processed = ROOT / cfg_all["paths"]["processed"]
    review_dir = processed / "review"
    review_dir.mkdir(exist_ok=True)

    rows = []
    for emg_path in sorted(processed.glob("ses-*/*_EMG.csv")):
        stem = emg_path.name[: -len("_EMG.csv")]
        df = pd.read_csv(emg_path)
        chans = [c for c in df.columns if c.startswith("EMG")]
        time = df["time"].to_numpy()
        sf = 1.0 / np.median(np.diff(time))
        segments, lab, env = label_run(
            df[chans].to_numpy().T, time, sf, df["bad"].to_numpy(),
            envelope_kw={**cfg["envelope"], "band": tuple(cfg["envelope"]["band"])},
            artifact_kw=cfg["artifact"], active_kw=cfg["active"])
        segments.to_csv(emg_path.with_name(f"{stem}_labels.csv"), index=False)

        dur = (segments.offset_s - segments.onset_s).groupby(segments.label).sum()
        moving = segments[segments.label == MOVING]
        row = {
            "run": stem,
            "duration_s": round(float(time[-1] - time[0]), 1),
            "n_bursts": len(moving),
            "moving_s": round(float(dur.get(MOVING, 0.0)), 1),
            "still_s": round(float(dur.get("still", 0.0)), 1),
            "artifact_s": round(float(dur.get(ARTIFACT, 0.0)), 1),
            "burst_median_s": round(float((moving.offset_s - moving.onset_s).median()), 2),
            "burst_interval_median_s": round(float(np.median(np.diff(moving.onset_s))), 2),
        }
        rows.append(row)
        print(", ".join(f"{k}={v}" for k, v in row.items()))
        if not args.no_plots:
            rest, burst = levels(env, sf, lab == ARTIFACT)
            review_plot(review_dir / f"{stem}.png", time, env, segments, rest, burst, cfg, stem)

    pd.DataFrame(rows).to_csv(processed / "labels_summary.csv", index=False)


if __name__ == "__main__":
    main()

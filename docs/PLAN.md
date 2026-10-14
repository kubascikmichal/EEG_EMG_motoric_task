# EEG / EMG hand-movement classification — findings & plan

Status: **step 1 (dataset build) implemented (2026-10-14); step 2a waits for the starting hand state.**
Original task: [prompts/init.md](../prompts/init.md).

> [!CAUTION]
> - **Git history rewrite (decision 6):** do not run it without the owner's explicit go-ahead at that moment.
> - **Labels (decision 1):** the labels come from the EMG envelope. Review the labelling method (step 2a) and
>   the circularity caveat with the owner before training any model.

Goal: classify hand movement phases (opening, opened, closing, closed) from
bioelectric signals only (IMU not used at all, not even for labels), comparing EEG-only, EMG-only and fusion,
and linear/statistical models vs. neural networks.

## 1. Dataset findings

- 3 subjects × 2 sessions, 12 recordings, ~50 min total.
  - MAXI = EEG, 32 ch, 250 Hz.
  - MINI = EMG, 8 ch, 500 Hz.
- Sources:
  - `dataset/csv/original/*.csv`: per device; filenames contain the date, device serial and UUID.
  - `dataset/raw/sub-*/ses-*/*.db`: SQLite, both devices in one file. Tables: `data_<uuid>` (numpy blobs `data`, `time`, plus `local_clock`), `meta_<uuid>` (channels, units, sf) and `annotations_<uuid>` (empty).
  - `dataset/csv/transformed/`: an earlier partial rename; to be replaced.
- All files are named `_eeg_`, including the EMG ones, and the EMG metadata labels its channels as type `EEG`.
- **No movement labels exist.** The annotation CSVs are empty or contain only a start timestamp.
- **Sync** (checked in step 1):
  - The CSV `time` column restarts at 0 per device. The CSV export also drops 3–366 leading samples of whichever device started first; otherwise the CSV values equal the `.db` values ×1e-6.
  - The `.db` `time` arrays are **not reliable for the EMG**: they are up to ±0.7 s away from the chunk arrival clock (`local_clock`), and compressed where filler blocks were inserted (see below). The earlier "1.36 s / 0.6–0.7 s start offsets" came from these arrays.
  - Timed from `local_clock` (shared host clock) and the real sample count, EEG and EMG start within **0–104 ms** of each other in every run.
  - Effective rates: EEG 249.99–250.01 Hz; EMG 499.84–500.01 Hz (up to 320 ppm slow, i.e. ~0.1 s over 5 min), so each run uses its fitted rate.
  - Arrival jitter is ~25–33 ms SD; the EMG clock wanders by up to ~70 ms within a run, which a 20 s running-median correction follows.
  - Not independently verified yet: an EMG-envelope vs. EEG high-frequency-power cross-correlation showed no coupling (|r| ≤ 0.25, scattered lags), so it can't confirm the timing either way. The remaining check is in step 2.
- **Naming collision:** sub-1 ses-S002 has two different recordings both named `run-001` (now run-01 and run-02).
- Sessions are interleaved within one evening; per the user, all sessions are the same task (see decision 2).
- **Signal quality:**
  - Scaling resolved: the `.db` is in µV as its metadata says; the CSV export is in V. EEG has a DC offset of ~15–26 mV (DC-coupled), no clipping.
  - EEG `contact_*` is always 0, so it carries no information. EMG `contact_*` is 1 except inside filler blocks.
  - EMG `Streaming=0` blocks (sub-1 ses-1 r2, sub-1 ses-2 r2, sub-2 ses-1 r3, sub-3 ses-2 r2) are **filler, not lost data**: always 100 zero samples, with the signal continuous across them, plus `Sample` counter jumps of −99. They are removed (4200 samples in total).
  - EMG clips at −2.25 V in one 25–31-sample stretch on one channel (sub-1 ses-1 r1, sub-1 ses-2 r2). It is marked in the `bad` column.
  - EMG at 500 Hz is limited to about 250 Hz (fine for envelope and time-domain features).
- The raw dataset with its original names is in git history (commit `2cc18d4`).

## 2. Decisions

| # | Question | Answer |
|---|----------|--------|
| 1 | Label source | **No protocol timing is available. Labels come from the EMG envelope** (step 2a). Caveat: the EMG-only model is then partly predicting its own labels, so its scores are inflated and the comparisons of EEG vs. EMG vs. fusion are biased toward EMG. Mitigations are in step 2a. |
| 2 | Session meaning | **All sessions are the same task/condition**; each recording = EEG + EMG of the same activity. Sessions are treated as equivalent repetitions (more data per subject, usable as CV folds). Note: alignment uses the `.db` chunk arrival clock (see section 1, Sync). |
| 3 | Duplicate run-001 (sub-1 ses-2) | **Number runs by session**: within each subject + session, runs are numbered in time order. Result: sub-1 ses-2 earlier → run-01, later → run-02. All other recordings already follow this rule, so their numbers don't change. |
| 4 | Output layout and format | **`dataset/processed/ses-XX/` + CSV**: `dataset/processed/ses-01/sub-01_ses-01_task-fusion_run-01_{EEG,EMG}.csv` plus a sidecar `.json` (sf, channels, start offset relative to the run). Implemented. |
| 5 | Existing `csv/original/`, `csv/transformed/` and `raw/` | **Add all three to `.gitignore`** and purge them from history (decision 6). They stay on disk, since the build still reads `raw/*.db`, but are no longer tracked. |
| 6 | Git history | **Rewrite history** (e.g. `git filter-repo`) to purge the original filenames and data, then force-push. Do this after step 1 is done and verified, and confirm with the owner right before running it. |
| 7 | IMU | **Not used at all**: not as model input, not for labels or label validation. Dropped in step 1. |

## 3. Plan

### Step 1 — dataset build (`scripts/01_build_dataset.py`) — DONE
Run: `python scripts/01_build_dataset.py [--check-csv]` (~30 s; settings in `config.yaml`, tests in `tests/`).
- Reads the `.db` files; `--check-csv` confirms all 24 device recordings match `csv/original` (they do).
- Removes the EMG filler blocks and timestamps every sample from the chunk arrival clock (`src/emgeeg/sync.py`).
- Output per run and modality: `dataset/processed/ses-XX/sub-XX_ses-XX_task-fusion_run-XX_{EEG,EMG}.csv` with `time` (s, shared by both devices, 0 = first sample of the earlier device), the signal channels in µV, and `bad` (EMG clipping ±50 ms). Contact, IMU, `Sample`, `Streaming` and battery columns are dropped.
- Sidecar `.json`: subject/session/task/run, channels, nominal and effective sf, start offset, removed filler, bad samples, timing fit quality. `dataset/processed/runs.csv` summarizes all runs.
- No date, time, serial, UUID or absolute clock value is written (checked with a grep).
- `.gitignore` covers `dataset/raw/`, `dataset/csv/original/` and `dataset/csv/transformed/`.
- Open: `dataset/processed/` (424 MB of CSV) is neither committed nor ignored yet.

### Step 2a — labels from the EMG envelope (script `scripts/02_make_labels.py`; review with the owner)
- **Envelope:** take the band-passed EMG (20–240 Hz, 50 Hz notch), rectify it, low-pass at about 3–6 Hz, and normalize each channel per run (e.g. to a percentile of its maximum).
- **Segmentation:** use a double threshold with hysteresis plus minimum durations to mark active and rest segments.
  - Opening and closing are the envelope bursts.
  - Opened and closed are the holds between bursts.
  - Telling opening from closing (and opened from closed): there is no extensor/flexor channel split, since all channels cover the forearm (owner, 2026-10-14). The movements are a repeated cycle, so the bursts alternate between opening and closing.
    1. Group the bursts into two clusters by their spatial pattern (per-channel envelope profile).
    2. Check that the clusters alternate. Where they don't (a missed or extra burst), flag the spot for manual review.
    3. Name each cluster opening or closing from the hand's starting state. **Open: does the hand start opened or closed in every run?**
    4. Each hold takes its name from the movement before it (opened after opening, closed after closing).
- **Review:** plot every run with its labels overlaid, fix them by hand where needed, and save `_labels.csv` (onset, offset, phase) next to each run.
- **Limiting circularity (decision 1):**
  - Smooth the labelling envelope much more heavily than the EMG classifier's features, and use its own thresholds, so the classifier can't simply learn the labelling rule back.
  - Report EMG-only results as an upper bound or a consistency check, not a fair competitor.
  - The main comparison is EEG vs. fusion.
  - Optional: split the EMG channels so that some make the labels and the others feed the classifier.

### Step 2 — notebook (thin; logic lives in the `src/` package)
- **Sync:** use the shared clock and a common window grid (e.g. 500 ms windows, 100 ms step). Each device keeps its native sampling rate.
- **EEG:** 50 Hz notch, 1–40 Hz band-pass, common average reference, sensorimotor channels (C3/Cz/C4/FC*/CP*), mu/beta ERD/ERS. ICA optional (no eye channel).
- **EMG:** 20–240 Hz band-pass, 50 Hz notch, RMS envelope, Hudgins time-domain features (MAV, WL, ZC, SSC).
- **Labels:** load them from step 2a and map them onto the window grid (each window takes the majority label; drop windows that straddle a transition, or mark them).
- **Fusion:**
  - early: concatenate the EEG and EMG features;
  - late: average or stack the two models' probabilities;
  - intermediate: a two-branch neural network.

### Step 3 — models and evaluation

| Type | EEG | EMG | Fusion |
|---|---|---|---|
| Linear / statistical | CSP+LDA, Riemannian tangent space + LR, band power + shrinkage LDA | time-domain features + LDA, linear SVM | concatenated features + LDA/LR, stacking |
| Neural networks | EEGNet, ShallowConvNet | 1D-CNN, CNN-LSTM | two-branch CNN |

- **Cross-validation:**
  - blocked splits within each run, with a gap so overlapping windows can't leak;
  - leave-one-run-out;
  - leave-one-subject-out.
- **Metrics:**
  - balanced accuracy, macro F1, Cohen's κ, confusion matrices;
  - empirical chance level from permutation tests;
  - for EMG detection: onset latency and false activations per minute.
- **Statistics:**
  - Friedman test with post-hoc tests across models;
  - Wilcoxon tests for EEG vs. EMG vs. fusion.

### Extra suggestions
- Add a 2-class task (moving vs. still) alongside the 4-class one. EEG can hardly separate the static "opened" and "closed" holds.
- Report a learning curve: neural networks will probably overfit with about 50 min of data.
- Artifact check: make sure the EEG classifier isn't really using EMG or motion artifacts (look at the feature topography and high-frequency power).
- Corticomuscular coherence: a sanity check for the sync and a possible fusion feature.
- Ablations: 3 vs. 32 EEG channels, window length, number of EMG channels.
- Pseudo-online evaluation: a sliding window with smoothing or majority voting, plus inference time per model.
- Reproducibility: one config file, fixed seeds, all results saved into one table.

### Proposed layout
```
src/emgeeg/   io.py  anonymize.py  sync.py  quality.py  preprocess.py  labels.py  features.py  models.py  evaluate.py
scripts/01_build_dataset.py
scripts/02_make_labels.py
notebooks/02_classification.ipynb
config.yaml
tests/
```
Stack: numpy, pandas, scipy, mne, scikit-learn, pyriemann, torch + braindecode.

## Next steps
1. ~~Implement step 1~~ done. Verify the sync in step 2 (EMG onsets vs. EEG movement-related potentials / beta ERD).
2. Owner: commit `dataset/processed/` or gitignore it?
3. Rewrite git history (decision 6), with the owner's go-ahead right before running it.
4. Owner: say whether the hand starts opened or closed in each run (channel layout and cycle answered: all channels, repeated cycle).
5. Step 2a: labels, reviewed with the owner on the plots.
6. Notebook: baselines first, then neural networks, then fusion.

# EEG / EMG hand-movement classification — findings & plan

Status: **consultation done, nothing implemented yet** (2026-10-13).
Original task: [prompts/init.md](../prompts/init.md).

Goal: classify hand movement phases (opening, opened, closing, closed) from
bioelectric signals only (no IMU), comparing EEG-only, EMG-only and fusion,
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
- **Sync:** the CSV `time` column restarts at 0 per device, so the EEG–EMG offset is lost.
  - The `.db` `time` arrays share one clock (LSL-like).
  - EEG vs. EMG start offsets on that clock are up to **1.36 s** (sub-1 ses-2 [time removed]), and 0.6–0.7 s in two of sub-3's runs.
  - → Build sync from the `.db` files, not the CSVs.
- **Naming collision:** sub-1 ses-S002 has two different recordings both named `run-001` ([time removed] and [time removed]).
- Sessions are interleaved within one evening, so a session is a condition, not a day.
- **Signal quality:**
  - The unit metadata says µV, but EEG values are about 0.02, which looks like volts. Verify the scaling.
  - EEG `contact_*` is always 0 and EMG `contact_*` is about 1. Either the conventions differ or the EEG flag isn't reported.
  - EMG saturates at −2.25 in some runs (sub-1 ses-1 r1, sub-1 ses-2 [time removed]) and has `Streaming=0` dropouts (sub-1 ses-1 r2, sub-1 ses-2 [time removed], sub-2 ses-1 r3, sub-3 ses-2 r2). These must be masked.
  - EMG at 500 Hz is limited to about 250 Hz (fine for envelope and time-domain features).
- The raw dataset with its original names is in git history (commit `2cc18d4`).

## 2. Decisions

| # | Question | Answer |
|---|----------|--------|
| 1 | Label source | "yes", interpreted as **(a) protocol timing**. **TODO: provide the cue sequence and phase durations** (and whether the timing is identical for every run). |
| 2 | Session meaning | "yes": **ses-1 = motor execution, ses-2 = kinesthetic motor imagery (KMI)**. **TODO: confirm the direction.** In KMI runs the EMG should be silent → EMG detection acts as a check for muscle activity leaking into imagery. |
| 3 | Duplicate run-001 (sub-1 ses-2) | open. Proposal: renumber by time ([time removed] → run-01, [time removed] → run-02). |
| 4 | Output layout and format | open. Proposal: `dataset/processed/ses-01/sub-01_ses-01_task-fusion_run-01_{EEG,EMG}.parquet` plus a sidecar `.json` (sf, channels, start offset relative to the run). CSV is possible if preferred. |
| 5 | Rewrite git history to remove original filenames | open. |

## 3. Plan

### Step 1 — dataset build (script `scripts/01_build_dataset.py`)
- Read the `.db` files and cross-check them against the CSVs.
- Keep the biosignal channels plus a quality mask built from the contact flags, saturation and `Streaming`. Drop the IMU, battery and sample columns.
- Anonymize: keep subject, session, task, run and EEG/EMG. Remove the date, time, serial, UUID and absolute clock values.
- Store the EEG–EMG offset relative to the run start in the sidecar `.json`.
- Organize the output into session folders.

### Step 2 — notebook (thin; logic lives in the `src/` package)
- **Sync:** use the shared clock and a common window grid (e.g. 500 ms windows, 100 ms step). Each device keeps its native sampling rate.
- **EEG:** 50 Hz notch, 1–40 Hz band-pass, common average reference, sensorimotor channels (C3/Cz/C4/FC*/CP*), mu/beta ERD/ERS. ICA optional (no eye channel).
- **EMG:** 20–240 Hz band-pass, 50 Hz notch, RMS envelope, Hudgins time-domain features (MAV, WL, ZC, SSC).
- **Labels** from the protocol timing (decision 1).
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
src/emgeeg/   io.py  anonymize.py  sync.py  preprocess.py  features.py  models.py  evaluate.py
scripts/01_build_dataset.py
notebooks/02_classification.ipynb
config.yaml
```
Stack: numpy, pandas, scipy, mne, scikit-learn, pyriemann, torch + braindecode.

## Next steps
1. Provide the protocol timing (decision 1) and confirm decisions 2–5.
2. Implement step 1, then verify the sync (EMG onsets vs. EEG movement-related potentials / coherence).
3. Notebook: baselines first, then neural networks, then fusion.

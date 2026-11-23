# Bearing Fault Diagnosis with the CWRU Dataset

Use vibration signals to classify a bearing as **normal** or affected by an **inner race**, **outer race**, or **ball** fault. This project prepares signal windows, extracts FFT features, and trains a Random Forest classifier.

## Contents

- [Get started](#get-started)
- [How the pipeline works](#how-the-pipeline-works)
- [Dataset layout](#dataset-layout)
- [Model and evaluation](#model-and-evaluation)
- [FAQ](#faq)
- [Ideas to try next](#ideas-to-try-next)

## Get started

Run these commands from the project folder, in order:

```bash
python inspect_cwru.py
python prepare_dataset.py
python train_baseline.py
python evaluate_splits.py
```

| Step | What it does |
| --- | --- |
| Inspect | Shows file details, signal statistics, and class counts. |
| Prepare | Normalizes signals and saves overlapping windows to `data/processed/cwru_windows.npz`. |
| Train | Trains a Random Forest and prints a classification report and confusion matrix. |
| Compare | Evaluates repeated one-file-per-class splits and leave-one-position-out splits. |

## How the pipeline works

```text
CWRU .mat files → normalized vibration signals → overlapping windows
                → log-magnitude FFT features → Random Forest predictions
```

Each window contains **2,048 samples**, with **50% overlap**. At a 12 kHz sample rate, one window covers about **170 ms**.

## Dataset layout

Place the raw `.mat` files in folders by condition:

```text
data/raw/
├── normal/
├── inner_race_fault/
├── outer_race/
└── ball_fault/
```

The project description expects four files per condition. Each file should contain drive-end vibration data sampled at 12 kHz.

<details>
<summary>What each condition means</summary>

- **Normal:** no seeded bearing fault.
- **Inner race fault:** damage on the bearing's inner ring.
- **Outer race fault:** damage on the bearing's outer ring.
- **Ball fault:** damage on a rolling element.

</details>

## Model and evaluation

The baseline uses a **300-tree Random Forest** with balanced class weights. It learns from log-magnitude FFT features, which represent frequency patterns in each vibration window.

The test split holds out complete bearing files. That checks whether the model can classify data from bearing units it did not train on. `evaluate_splits.py` also repeats the one-file-per-class split with 10 random seeds and checks each file position in turn.

> Accuracy depends on the files and split. Treat the expected **>95%** accuracy as a project target, not a guarantee.

<details>
<summary>Why split by file?</summary>

Windows from one recording are related. If windows from the same file appear in both training and test sets, the test score can look better than performance on a new bearing. Holding out whole files gives a more realistic check of generalization.

</details>

## FAQ

<details>
<summary>How do I change the window size or overlap?</summary>

Edit `WINDOW_SIZE` and `STRIDE` in `prepare_dataset.py`, then run the preparation and training steps again.

</details>

<details>
<summary>How do I predict on new vibration data?</summary>

Apply the same normalization and windowing, extract the same FFT features, and pass them to the trained model. See `train_baseline.py` for the feature and prediction pattern.

</details>

<details>
<summary>Where are the processed windows saved?</summary>

`data/processed/cwru_windows.npz`

</details>


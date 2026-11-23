"""
Dataset preparation for bearing fault classification.

Converts raw CWRU signals into ML-ready format: loads, normalizes, windows, 
and saves as compressed NumPy archive.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/cwru_windows.npz")

LABEL_TO_ID = {
    "normal": 0,
    "inner_race_fault": 1,
    "outer_race": 2,
    "ball_fault": 3,
}

WINDOW_SIZE = 2048
STRIDE = 1024


def find_drive_end_signal(mat_dict):
    """Extract the drive-end vibration signal from a CWRU .mat file."""
    keys = [key for key in mat_dict if key.endswith("DE_time")]
    if not keys:
        raise KeyError("No drive-end vibration key ending in 'DE_time' found.")
    return mat_dict[keys[0]].reshape(-1).astype(np.float32)


def normalize(signal):
    """Standardize signal to zero-mean, unit-variance."""
    signal = signal - signal.mean()
    std = signal.std()
    if std > 0:
        signal = signal / std
    return signal


def make_windows(signal):
    """Create overlapping windows from a continuous signal."""
    if len(signal) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)

    starts = range(0, len(signal) - WINDOW_SIZE + 1, STRIDE)
    windows = [signal[start : start + WINDOW_SIZE] for start in starts]
    return np.stack(windows).astype(np.float32)


def main():
    """Process all raw bearing data files and create the ML dataset."""
    all_windows = []
    all_labels = []
    all_groups = []
    file_counts = {}
    window_counts = {}

    files = sorted(RAW_DIR.glob("*/*.mat"))
    if not files:
        raise FileNotFoundError("No .mat files found under data/raw/<class_name>/")

    for path in files:
        label_name = path.parent.name
        if label_name not in LABEL_TO_ID:
            raise ValueError(f"Unknown label folder: {label_name}")

        mat_dict = loadmat(path)
        signal = normalize(find_drive_end_signal(mat_dict))
        windows = make_windows(signal)
        label_id = LABEL_TO_ID[label_name]

        all_windows.append(windows)
        all_labels.append(np.full(len(windows), label_id, dtype=np.int64))
        all_groups.append(np.full(len(windows), path.as_posix()))

        file_counts[label_name] = file_counts.get(label_name, 0) + 1
        window_counts[label_name] = window_counts.get(label_name, 0) + len(windows)

    x = np.concatenate(all_windows)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        x=x,
        y=y,
        groups=groups,
        label_names=np.array(
            ["normal", "inner_race_fault", "outer_race", "ball_fault"]
        ),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print()
    print("Files per class:")
    for label in sorted(file_counts):
        print(f"  {label}: {file_counts[label]}")
    print()
    print("Windows per class:")
    for label in sorted(window_counts):
        print(f"  {label}: {window_counts[label]}")


if __name__ == "__main__":
    main()

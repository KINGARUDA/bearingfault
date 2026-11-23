"""
Data inspection utility for CWRU bearing dataset.

Examine raw .mat files and display file locations, signal statistics, 
class distribution, and signal key names.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


RAW_DIR = Path("data/raw")

VALID_LABELS = {
    "normal",
    "inner_race_fault",
    "ball_fault",
    "outer_race",
}


def find_drive_end_signal(mat_dict):
    """Extract the drive-end (DE) vibration signal from a CWRU .mat file."""
    keys = [key for key in mat_dict if key.endswith("DE_time")]
    if not keys:
        raise KeyError("No drive-end vibration key ending in 'DE_time' found.")
    signal = mat_dict[keys[0]].reshape(-1).astype(np.float32)
    return keys[0], signal


def main():
    """Scan all raw .mat files and display their properties."""
    files = sorted(RAW_DIR.glob("*/*.mat"))
    if not files:
        print("No .mat files found under data/raw/<class_name>/")
        return

    unknown_labels = sorted(
        {path.parent.name for path in files if path.parent.name not in VALID_LABELS}
    )
    if unknown_labels:
        print("Unknown class folders:")
        for label in unknown_labels:
            print(f"  - {label}")
        print()
        print(f"Expected folders: {', '.join(sorted(VALID_LABELS))}")
        return

    print(f"Found {len(files)} .mat files.")
    print()

    counts = {}
    for path in files:
        label = path.parent.name
        counts[label] = counts.get(label, 0) + 1

        mat_dict = loadmat(path)
        key, signal = find_drive_end_signal(mat_dict)

        print(path.as_posix())
        print(f"  label: {label}")
        print(f"  signal key: {key}")
        print(f"  samples: {signal.shape[0]}")
        print(f"  mean: {signal.mean():.6f}")
        print(f"  std: {signal.std():.6f}")
        print()

    print("Class counts:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()

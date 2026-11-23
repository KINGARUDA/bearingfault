"""
Compare different train-test split strategies for bearing fault classification.

Evaluates two approaches:
1. Repeated random one-file-per-class holdout (10 seeds)
2. Leave-one-position split (systematically hold out file at position 0-3)
"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "data/processed/cwru_windows.npz"


def fft_features(x):
    """Extract frequency-domain features using Fast Fourier Transform."""
    spectrum = np.fft.rfft(x, axis=1)
    magnitude = np.abs(spectrum)
    return np.log1p(magnitude).astype(np.float32)


def make_model():
    """Create a Random Forest pipeline with standard configuration."""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )


def split_one_file_per_class(y, groups, seed):
    """Create train/test split with one random file per class in test set."""
    rng = np.random.default_rng(seed)
    test_groups = []

    for label_id in sorted(np.unique(y)):
        label_groups = np.unique(groups[y == label_id])
        test_groups.append(rng.choice(label_groups))

    test_groups = np.array(test_groups)
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    train_idx = np.flatnonzero(~np.isin(groups, test_groups))
    return train_idx, test_idx


def file_stem(group):
    """Extract filename without path/extension."""
    return Path(str(group)).stem


def split_leave_one_position(groups, position):
    """
    Hold out the file at a specific position within each class.
    
    Position 0 = first file, Position 1 = second file, etc.
    """
    groups_by_label = {}
    for group in np.unique(groups):
        label = Path(str(group)).parent.name
        groups_by_label.setdefault(label, []).append(group)

    test_groups = []
    for label, label_groups in groups_by_label.items():
        sorted_groups = sorted(label_groups, key=file_stem)
        if position >= len(sorted_groups):
            raise ValueError(f"Class {label} has no file at position {position}")
        test_groups.append(sorted_groups[position])

    test_groups = np.array(test_groups)
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    train_idx = np.flatnonzero(~np.isin(groups, test_groups))
    return train_idx, test_idx


def evaluate_split(x_features, y, groups, label_names, train_idx, test_idx):
    """Train model on train set and evaluate on test set."""
    model = make_model()
    model.fit(x_features[train_idx], y[train_idx])
    y_pred = model.predict(x_features[test_idx])
    return accuracy_score(y[test_idx], y_pred), y_pred


def print_groups(title, groups, indices):
    """Pretty-print which files are in a given set."""
    print(title)
    for group in np.unique(groups[indices]):
        print(f"  {group}")
    print()


def main():
    """Run split strategy evaluations."""
    data = np.load(DATASET_PATH, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    groups = data["groups"]
    label_names = data["label_names"]
    x_features = fft_features(x)

    print("Repeated one-file-per-class holdout (10 random seeds)")
    print("-" * 60)
    accuracies = []
    for seed in range(10):
        train_idx, test_idx = split_one_file_per_class(y, groups, seed)
        accuracy, _ = evaluate_split(x_features, y, groups, label_names, train_idx, test_idx)
        accuracies.append(accuracy)
        test_files = ", ".join(str(group) for group in np.unique(groups[test_idx]))
        print(f"  seed {seed}: accuracy={accuracy:.4f}; test={test_files}")

    print()
    print(
        "Accuracy summary: "
        f"min={np.min(accuracies):.4f}, "
        f"mean={np.mean(accuracies):.4f}, "
        f"max={np.max(accuracies):.4f}"
    )
    print()

    print("Leave-one-position-out split")
    print("-" * 60)
    for position in range(4):
        train_idx, test_idx = split_leave_one_position(groups, position)
        accuracy, y_pred = evaluate_split(
            x_features, y, groups, label_names, train_idx, test_idx
        )
        print(f"Position {position}: accuracy={accuracy:.4f}")
        print_groups("Test files:", groups, test_idx)
        print(classification_report(y[test_idx], y_pred, target_names=label_names))
        print("Confusion matrix:")
        print(confusion_matrix(y[test_idx], y_pred))
        print()


if __name__ == "__main__":
    main()

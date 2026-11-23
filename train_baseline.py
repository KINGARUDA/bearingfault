"""
Train a baseline Random Forest model for bearing fault classification.

Loads data, extracts FFT features, splits with one-file-per-class holdout,
trains Random Forest, and evaluates with classification report and confusion matrix.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "data/processed/cwru_windows.npz"


def fft_features(x):
    """Extract frequency-domain features using Fast Fourier Transform."""
    spectrum = np.fft.rfft(x, axis=1)
    magnitude = np.abs(spectrum)
    return np.log1p(magnitude).astype(np.float32)


def split_one_file_per_class(y, groups):
    """
    Create train/test split with one file per class in the test set.
    
    Ensures model generalizes to unseen bearing units.
    """
    rng = np.random.default_rng(42)
    train_groups = []
    test_groups = []

    for label_id in sorted(np.unique(y)):
        label_groups = np.unique(groups[y == label_id])
        test_group = rng.choice(label_groups)
        test_groups.append(test_group)
        train_groups.extend(group for group in label_groups if group != test_group)

    test_groups = np.array(test_groups)
    train_groups = np.array(train_groups)

    train_idx = np.flatnonzero(np.isin(groups, train_groups))
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    return train_idx, test_idx


def main():
    """Load data, train model, and evaluate on test set."""
    data = np.load(DATASET_PATH, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    groups = data["groups"]
    label_names = data["label_names"]

    x_features = fft_features(x)
    train_idx, test_idx = split_one_file_per_class(y, groups)

    x_train = x_features[train_idx]
    x_test = x_features[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])

    print("Train files:")
    for group in train_groups:
        print(f"  {group}")
    print()
    print("Test files:")
    for group in test_groups:
        print(f"  {group}")
    print()

    missing_test_classes = sorted(set(np.unique(y)) - set(np.unique(y_test)))
    if missing_test_classes:
        missing = ", ".join(label_names[idx] for idx in missing_test_classes)
        raise ValueError(f"Test split is missing classes: {missing}")

    missing_train_classes = sorted(set(np.unique(y)) - set(np.unique(y_train)))
    if missing_train_classes:
        missing = ", ".join(label_names[idx] for idx in missing_train_classes)
        raise ValueError(f"Train split is missing classes: {missing}")

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()

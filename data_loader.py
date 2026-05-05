import os
import numpy as np
import mne


def load_subject(file_path, tmin=0.5, tmax=3.5, apply_filter=True):
    """
    Load a single subject file and return epochs
    """
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Optional bandpass filtering (IMPORTANT for EEG)
    if apply_filter:
        raw.filter(8., 30., fir_design='firwin')

    # BCI-IV 2a annotation codes:
    # '769' = Left hand, '770' = Right hand
    fixed_mapping = {'769': 1, '770': 2}
    events, _ = mne.events_from_annotations(raw, event_id=fixed_mapping)

    target_events = {'left': 1, 'right': 2}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        event_repeated='drop'
    )

    return epochs


def load_data(data_path, train_subjects=6, tmin=0.5, tmax=3.5):
    """
    Load dataset and split into train/test (cross-subject)
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith("T.gdf")])

    train_files = files[:train_subjects]
    test_files = files[train_subjects:]

    train_epochs = []
    test_epochs = []

    print("Loading TRAIN subjects...")
    for f in train_files:
        print(f"  {f}")
        ep = load_subject(os.path.join(data_path, f), tmin=tmin, tmax=tmax)
        train_epochs.append(ep)

    print("Loading TEST subjects...")
    for f in test_files:
        print(f"  {f}")
        ep = load_subject(os.path.join(data_path, f), tmin=tmin, tmax=tmax)
        test_epochs.append(ep)

    # Concatenate
    train_epochs = mne.concatenate_epochs(train_epochs)
    test_epochs = mne.concatenate_epochs(test_epochs)

    # Extract data
    X_train = train_epochs.get_data()
    y_train = train_epochs.events[:, -1]

    X_test = test_epochs.get_data()
    y_test = test_epochs.events[:, -1]

    # Convert labels: right (code 2) -> 1, left (code 1) -> 0
    y_train = (y_train == 2).astype(int)
    y_test = (y_test == 2).astype(int)

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    return X_train, y_train, X_test, y_test
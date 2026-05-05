# preprocess.py

import numpy as np


# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize(X_train, X_test):
    """
    Normalize using training statistics
    """
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(np.float32), X_test.astype(np.float32)


# -------------------------------
# CHANNEL DIMENSION (FOR CNN)
# -------------------------------
def add_channel_dim(X):
    """
    (N, C, T) → (N, C, T, 1)
    """
    return X[..., np.newaxis]


# -------------------------------
# SIMPLE DATA AUGMENTATION
# -------------------------------
def augment_noise(X, noise_level=0.01):
    """
    Add Gaussian noise
    """
    noise = noise_level * np.random.randn(*X.shape)
    return X + noise


def augment_time_shift(X, max_shift=50):
    """
    Random time shift (circular)
    """
    X_shifted = np.empty_like(X)

    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift)
        X_shifted[i] = np.roll(X[i], shift, axis=-1)

    return X_shifted


def augment_amplitude_scaling(X, scale_range=(0.9, 1.1)):
    """
    Random amplitude scaling
    """
    scales = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1, 1))
    return X * scales


# -------------------------------
# AUGMENTATION PIPELINE
# -------------------------------
def apply_augmentation(X, use_noise=True, use_shift=True, use_scaling=True):
    """
    Apply multiple augmentations
    """
    X_aug = X.copy()

    if use_noise:
        X_aug = augment_noise(X_aug)

    if use_shift:
        X_aug = augment_time_shift(X_aug)

    if use_scaling:
        X_aug = augment_amplitude_scaling(X_aug)

    return X_aug


# -------------------------------
# BAND SELECTION (OPTIONAL)
# -------------------------------
def select_time_window(X, start=0, end=None):
    """
    Select part of the time window
    Useful for experiments (e.g., 0.5–3.5 sec)
    """
    return X[:, :, start:end]


# -------------------------------
# FULL PREPROCESS PIPELINE
# -------------------------------
def preprocess_pipeline(X_train, X_test, apply_aug=False):
    """
    Full preprocessing pipeline
    """

    # Normalize
    X_train, X_test = normalize(X_train, X_test)

    # Optional augmentation (ONLY on training)
    if apply_aug:
        X_train = apply_augmentation(X_train)

    return X_train, X_test
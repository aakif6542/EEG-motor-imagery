# preprocessing/pipeline.py
# ============================================================
# Unified preprocessing pipeline for EEG data.
#
# PRESERVED: Original preprocessing ideas (normalization,
#   Gaussian noise, time shift, amplitude scaling).
# IMPROVED: Config-driven, handles different channel counts
#   and sample rates, reusable across all datasets.
# ============================================================

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------
def normalize(X_train, X_test):
    """
    Channel-wise normalization using training statistics.
    Preserved from original preprocess.py.
    """
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(np.float32), X_test.astype(np.float32)


# ---------------------------------------------------------------
# CHANNEL DIMENSION (FOR CNN / EEGNet)
# ---------------------------------------------------------------
def add_channel_dim(X):
    """
    Add trailing channel dimension: (N, C, T) → (N, C, T, 1).
    Required for 2D convolution in Keras.
    """
    return X[..., np.newaxis]


# ---------------------------------------------------------------
# DATA AUGMENTATION (TRAINING ONLY)
# ---------------------------------------------------------------
def augment_noise(X, noise_level=0.01):
    """Add Gaussian noise. Preserved from original."""
    noise = noise_level * np.random.randn(*X.shape)
    return X + noise


def augment_time_shift(X, max_shift=50):
    """Random circular time shift. Preserved from original."""
    X_shifted = np.empty_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift)
        X_shifted[i] = np.roll(X[i], shift, axis=-1)
    return X_shifted


def augment_amplitude_scaling(X, scale_range=(0.9, 1.1)):
    """Random amplitude scaling. Preserved from original."""
    scales = np.random.uniform(
        scale_range[0], scale_range[1],
        size=(X.shape[0], 1, 1)
    )
    return X * scales


def apply_augmentation(X, use_noise=True, use_shift=True,
                       use_scaling=True, noise_level=0.01,
                       max_shift=50, scale_range=(0.9, 1.1)):
    """
    Apply multiple augmentations to training data.
    Preserved from original with added configurability.
    """
    X_aug = X.copy()

    if use_noise:
        X_aug = augment_noise(X_aug, noise_level=noise_level)

    if use_shift:
        X_aug = augment_time_shift(X_aug, max_shift=max_shift)

    if use_scaling:
        X_aug = augment_amplitude_scaling(X_aug, scale_range=scale_range)

    return X_aug


# ---------------------------------------------------------------
# TIME WINDOW SELECTION
# ---------------------------------------------------------------
def select_time_window(X, start=0, end=None):
    """
    Select a sub-window of the time dimension.
    Preserved from original.
    """
    return X[:, :, start:end]


# ---------------------------------------------------------------
# FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------------
def preprocess_pipeline(X_train, X_test, config=None):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    X_train : np.ndarray
        Training data (N, C, T).
    X_test : np.ndarray
        Test data (N, C, T).
    config : PreprocessConfig, optional
        Preprocessing configuration. If None, uses defaults.

    Returns
    -------
    X_train, X_test : preprocessed arrays
    """

    # Use defaults if no config provided
    do_normalize = True
    do_augment = False
    aug_noise = True
    aug_shift = True
    aug_scaling = True
    noise_level = 0.01
    max_shift = 50
    scale_range = (0.9, 1.1)

    if config is not None:
        do_normalize = config.normalize
        do_augment = config.augment
        aug_noise = config.augment_noise
        aug_shift = config.augment_time_shift
        aug_scaling = config.augment_amplitude_scaling
        noise_level = config.augment_noise_level
        max_shift = config.augment_max_shift
        scale_range = config.augment_scale_range

    # Normalize
    if do_normalize:
        X_train, X_test = normalize(X_train, X_test)

    # Augmentation (training data only)
    if do_augment:
        X_train = apply_augmentation(
            X_train,
            use_noise=aug_noise,
            use_shift=aug_shift,
            use_scaling=aug_scaling,
            noise_level=noise_level,
            max_shift=max_shift,
            scale_range=scale_range,
        )

    return X_train, X_test

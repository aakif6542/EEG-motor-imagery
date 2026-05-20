# models/base_model.py
# ============================================================
# Abstract base class for all EEG classification models.
# Provides a unified interface for training, prediction,
# evaluation, and persistence. Designed to be extended by
# both classical ML (CSP+SVM) and deep learning models.
# ============================================================

from abc import ABC, abstractmethod
import numpy as np
import os


class BaseModel(ABC):
    """
    Abstract base class for EEG classification models.

    All models must implement: build, fit, predict, evaluate,
    save, load. This ensures the experiment runner can treat
    every model identically.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        ...

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.

        Returns training history (dict or keras History) for DL models,
        or None for classical models.
        """
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predicted class labels (integer array)."""
        ...

    @abstractmethod
    def evaluate(self, X_test, y_test) -> float:
        """Return accuracy score."""
        ...

    @abstractmethod
    def save(self, dirpath: str):
        """Save model to directory."""
        ...

    @abstractmethod
    def load(self, dirpath: str):
        """Load model from directory."""
        ...

    def needs_channel_dim(self) -> bool:
        """
        Whether the model needs a trailing channel dimension.
        Override to True for CNN/EEGNet (4D input).
        Classical models use 3D input (N, C, T).
        """
        return False

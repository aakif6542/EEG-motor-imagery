# datasets/base_dataset.py
# ============================================================
# Abstract base class for all EEG dataset loaders.
# Defines a unified interface that every dataset must implement,
# ensuring consistent data format across BNCI2014001, PhysioNet,
# and Cho2017.
# ============================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class EEGDataBundle:
    """
    Standardized container for loaded EEG data.

    All dataset loaders must return data in this format,
    regardless of the original dataset structure.
    """
    X_train: np.ndarray          # (N_train, C, T) — trials × channels × timepoints
    y_train: np.ndarray          # (N_train,)      — integer class labels
    X_test: np.ndarray           # (N_test, C, T)
    y_test: np.ndarray           # (N_test,)

    n_channels: int              # Number of EEG channels
    n_timepoints: int            # Number of time samples per trial
    sfreq: float                 # Sampling frequency (Hz)

    class_names: List[str]       # Human-readable class labels
    n_classes: int = 2           # Number of classes

    subject_ids_train: List[int] = field(default_factory=list)
    subject_ids_test: List[int] = field(default_factory=list)

    dataset_name: str = ""       # Name of the dataset for logging

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency after initialization."""
        assert self.X_train.ndim == 3, \
            f"X_train must be 3D (N, C, T), got shape {self.X_train.shape}"
        assert self.X_test.ndim == 3, \
            f"X_test must be 3D (N, C, T), got shape {self.X_test.shape}"
        assert len(self.y_train) == self.X_train.shape[0], \
            f"y_train length ({len(self.y_train)}) != X_train trials ({self.X_train.shape[0]})"
        assert len(self.y_test) == self.X_test.shape[0], \
            f"y_test length ({len(self.y_test)}) != X_test trials ({self.X_test.shape[0]})"

    def summary(self) -> str:
        """Return a human-readable summary of the data bundle."""
        lines = [
            f"{'='*50}",
            f"  Dataset: {self.dataset_name}",
            f"{'='*50}",
            f"  Channels:    {self.n_channels}",
            f"  Timepoints:  {self.n_timepoints}",
            f"  Sample rate: {self.sfreq} Hz",
            f"  Classes:     {self.class_names} ({self.n_classes})",
            f"  Train:       {self.X_train.shape[0]} trials (subjects: {self.subject_ids_train})",
            f"  Test:        {self.X_test.shape[0]} trials (subjects: {self.subject_ids_test})",
            f"  X_train:     {self.X_train.shape}",
            f"  X_test:      {self.X_test.shape}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class BaseDataset(ABC):
    """
    Abstract base class for all EEG dataset loaders.

    Subclasses must implement:
        - load() → EEGDataBundle
        - dataset_name (property)
        - n_subjects (property)

    The load() method should use MOABB to fetch data and return
    a standardized EEGDataBundle.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human-readable name for this dataset."""
        ...

    @property
    @abstractmethod
    def n_subjects(self) -> int:
        """Total number of subjects in the dataset."""
        ...

    @abstractmethod
    def load(
        self,
        train_subjects: Optional[List[int]] = None,
        test_subjects: Optional[List[int]] = None,
        tmin: float = 0.5,
        tmax: float = 3.5,
        bandpass_low: float = 8.0,
        bandpass_high: float = 30.0,
        resample_freq: Optional[float] = 128.0,
    ) -> EEGDataBundle:
        """
        Load and preprocess the dataset.

        Parameters
        ----------
        train_subjects : list of int, optional
            Subject IDs for training. If None, uses default split.
        test_subjects : list of int, optional
            Subject IDs for testing. If None, uses default split.
        tmin : float
            Start time of epoch (seconds after event onset).
        tmax : float
            End time of epoch (seconds after event onset).
        bandpass_low : float
            Low cutoff for bandpass filter (Hz).
        bandpass_high : float
            High cutoff for bandpass filter (Hz).
        resample_freq : float, optional
            Resample to this frequency. None = no resampling.

        Returns
        -------
        EEGDataBundle
            Standardized data container.
        """
        ...

    def get_default_split(self, train_ratio: float = 0.7):
        """
        Get default train/test subject split.

        Parameters
        ----------
        train_ratio : float
            Fraction of subjects for training.

        Returns
        -------
        train_subjects, test_subjects : lists of int
        """
        all_subjects = list(range(1, self.n_subjects + 1))
        n_train = max(1, int(len(all_subjects) * train_ratio))
        return all_subjects[:n_train], all_subjects[n_train:]

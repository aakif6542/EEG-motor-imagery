# datasets/bnci2014001.py
# ============================================================
# MOABB-based loader for BCI Competition IV Dataset 2a
# (BNCI2014001). 9 subjects, 4-class motor imagery.
#
# This replaces the original data_loader.py which required
# manual GDF file management. MOABB handles downloading and
# caching automatically.
# ============================================================

import numpy as np
import mne
from typing import List, Optional

from datasets.base_dataset import BaseDataset, EEGDataBundle


class BNCI2014001Dataset(BaseDataset):
    """
    BCI Competition IV Dataset 2a (BNCI2014001).

    - 9 subjects
    - 4 classes: left hand, right hand, feet, tongue
    - 22 EEG channels + 3 EOG
    - 250 Hz sample rate
    - We use only left/right for binary classification
      (consistent with original project)
    """

    @property
    def dataset_name(self) -> str:
        return "BNCI2014001"

    @property
    def n_subjects(self) -> int:
        return 9

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
        """Load BNCI2014001 via MOABB with cross-subject split."""

        from moabb.datasets import BNCI2014_001
        from utils.logging_config import get_logger
        logger = get_logger()

        # Default split: 6 train, 3 test (matching original project)
        if train_subjects is None:
            train_subjects = [1, 2, 3, 4, 5, 6]
        if test_subjects is None:
            test_subjects = [7, 8, 9]

        logger.info(f"Loading dataset: {self.dataset_name}")
        logger.info(f"Train subjects: {train_subjects}")
        logger.info(f"Test subjects:  {test_subjects}")

        dataset = BNCI2014_001()

        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        # --- Load training subjects ---
        for subj in train_subjects:
            logger.info(f"Subject {subj} processing...")
            X_subj, y_subj = self._load_subject(
                dataset, subj, tmin, tmax,
                bandpass_low, bandpass_high, resample_freq
            )
            if X_subj is not None:
                X_train_list.append(X_subj)
                y_train_list.append(y_subj)

        # --- Load test subjects ---
        for subj in test_subjects:
            logger.info(f"Subject {subj} processing...")
            X_subj, y_subj = self._load_subject(
                dataset, subj, tmin, tmax,
                bandpass_low, bandpass_high, resample_freq
            )
            if X_subj is not None:
                X_test_list.append(X_subj)
                y_test_list.append(y_subj)

        # --- Concatenate ---
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        logger.info(f"Dataset {self.dataset_name} loaded successfully (Train: {X_train.shape}, Test: {X_test.shape})")

        return EEGDataBundle(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_channels=X_train.shape[1],
            n_timepoints=X_train.shape[2],
            sfreq=resample_freq if resample_freq else 250.0,
            class_names=["left_hand", "right_hand"],
            n_classes=2,
            subject_ids_train=train_subjects,
            subject_ids_test=test_subjects,
            dataset_name=self.dataset_name,
        )

    def _load_subject(
        self, dataset, subject_id, tmin, tmax,
        bandpass_low, bandpass_high, resample_freq
    ):
        """Load a single subject's data from the MOABB dataset object."""
        from utils.logging_config import get_logger
        logger = get_logger()

        try:
            # MOABB returns dict: {session_name: {run_name: Raw}}
            subject_data = dataset.get_data(subjects=[subject_id])

            all_epochs = []

            for session_name, session_runs in subject_data[subject_id].items():
                for run_name, raw in session_runs.items():

                    raw = raw.copy().load_data()

                    # Select only EEG channels (drop EOG)
                    raw.pick_types(eeg=True)

                    # Bandpass filter (mu + beta bands)
                    raw.filter(bandpass_low, bandpass_high, fir_design='firwin')

                    # Resample if requested
                    if resample_freq and raw.info['sfreq'] != resample_freq:
                        raw.resample(resample_freq)

                    # Extract events from annotations
                    events, event_id = mne.events_from_annotations(raw)

                    # Map to left/right only
                    # MOABB BNCI2014_001 annotations: left_hand, right_hand, feet, tongue
                    target_event_id = {}
                    for key, val in event_id.items():
                        if 'left' in key.lower():
                            target_event_id[key] = val
                        elif 'right' in key.lower():
                            target_event_id[key] = val

                    if not target_event_id:
                        continue

                    epochs = mne.Epochs(
                        raw, events,
                        event_id=target_event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=None,
                        preload=True,
                        event_repeated='drop',
                        verbose=False,
                    )

                    if len(epochs) > 0:
                        all_epochs.append(epochs)

            if not all_epochs:
                logger.warning(f"No valid epochs for subject {subject_id}")
                return None, None

            combined = mne.concatenate_epochs(all_epochs, verbose=False)
            X = combined.get_data(copy=False)   # (N, C, T)
            y = combined.events[:, -1]

            # Convert to binary: find unique event IDs and map
            unique_events = sorted(np.unique(y))
            if len(unique_events) >= 2:
                # Map first class → 0, second class → 1
                label_map = {unique_events[0]: 0, unique_events[1]: 1}
                y = np.array([label_map[val] for val in y])

            logger.info(f"Subject {subject_id} loaded ({X.shape[0]} trials)")

            return X, y

        except Exception as e:
            logger.error(f"Error loading subject {subject_id}: {e}")
            return None, None

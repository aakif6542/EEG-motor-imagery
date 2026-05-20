# datasets/physionet_mi.py
# ============================================================
# MOABB-based loader for PhysioNet Motor Imagery dataset.
# 109 subjects, 64 EEG channels, 160 Hz.
# ============================================================

import numpy as np
import mne
from typing import List, Optional
from datasets.base_dataset import BaseDataset, EEGDataBundle


class PhysionetMIDataset(BaseDataset):
    """
    PhysioNet EEG Motor Movement/Imagery Dataset.
    - 109 subjects, 64 channels, 160 Hz
    - Binary: left fist vs right fist
    """

    @property
    def dataset_name(self) -> str:
        return "PhysionetMI"

    @property
    def n_subjects(self) -> int:
        return 109

    def load(self, train_subjects=None, test_subjects=None,
             tmin=0.5, tmax=3.5, bandpass_low=8.0, bandpass_high=30.0,
             resample_freq=128.0) -> EEGDataBundle:

        from moabb.datasets import PhysionetMI
        from utils.logging_config import get_logger
        logger = get_logger()

        if train_subjects is None:
            train_subjects = list(range(1, 11))
        if test_subjects is None:
            test_subjects = list(range(11, 16))

        logger.info(f"Loading dataset: {self.dataset_name} via MOABB")
        dataset = PhysionetMI()

        X_tr, y_tr = self._load_subjects(dataset, train_subjects, tmin, tmax,
                                          bandpass_low, bandpass_high, resample_freq)
        X_te, y_te = self._load_subjects(dataset, test_subjects, tmin, tmax,
                                          bandpass_low, bandpass_high, resample_freq)

        logger.info(f"Dataset {self.dataset_name} loaded successfully (Train: {X_tr.shape}, Test: {X_te.shape})")

        return EEGDataBundle(
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            n_channels=X_tr.shape[1], n_timepoints=X_tr.shape[2],
            sfreq=resample_freq or 160.0,
            class_names=["left_hand", "right_hand"], n_classes=2,
            subject_ids_train=train_subjects, subject_ids_test=test_subjects,
            dataset_name=self.dataset_name,
        )

    def _load_subjects(self, dataset, subjects, tmin, tmax, bl, bh, resample):
        from utils.logging_config import get_logger
        logger = get_logger()
        Xs, ys = [], []
        for s in subjects:
            logger.info(f"Subject {s} processing...")
            X, y = self._load_one(dataset, s, tmin, tmax, bl, bh, resample)
            if X is not None:
                Xs.append(X); ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    def _load_one(self, dataset, sid, tmin, tmax, bl, bh, resample):
        from utils.logging_config import get_logger
        logger = get_logger()
        try:
            data = dataset.get_data(subjects=[sid])
            all_epochs = []
            for sess in data[sid].values():
                for raw in sess.values():
                    raw = raw.copy().load_data()
                    raw.pick_types(eeg=True)
                    raw.filter(bl, bh, fir_design='firwin')
                    if resample and raw.info['sfreq'] != resample:
                        raw.resample(resample)
                    events, eid = mne.events_from_annotations(raw)
                    tgt = {k: v for k, v in eid.items()
                           if ('left' in k.lower() or 'right' in k.lower())
                           and ('hand' in k.lower() or 'fist' in k.lower())}
                    if not tgt:
                        continue
                    ep = mne.Epochs(raw, events, event_id=tgt, tmin=tmin,
                                    tmax=tmax, baseline=None, preload=True,
                                    event_repeated='drop', verbose=False)
                    if len(ep) > 0:
                        all_epochs.append(ep)
            if not all_epochs:
                return None, None
            combined = mne.concatenate_epochs(all_epochs, verbose=False)
            X = combined.get_data(copy=False)
            y = combined.events[:, -1]
            uq = sorted(np.unique(y))
            if len(uq) >= 2:
                lm = {uq[0]: 0, uq[1]: 1}
                y = np.array([lm.get(v, -1) for v in y])
                valid = y >= 0
                X, y = X[valid], y[valid]
            logger.info(f"Subject {sid} loaded ({X.shape[0]} trials)")
            return X, y
        except Exception as e:
            logger.error(f"Error subject {sid}: {e}")
            return None, None

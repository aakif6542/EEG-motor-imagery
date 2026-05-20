# models/csp_svm.py
# ============================================================
# CSP + SVM pipeline for EEG motor imagery classification.
#
# PRESERVED: Core architecture from original project.
# IMPROVED: Inherits BaseModel, adds save/load persistence.
# ============================================================

import os
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from models.base_model import BaseModel


class CSPSVMModel(BaseModel):
    """
    Common Spatial Pattern (CSP) + Support Vector Machine (SVM).

    Classical approach: CSP learns spatial filters that maximize
    variance differences between classes, SVM classifies the
    log-variance features.
    """

    def __init__(self, n_components=4, kernel='linear'):
        """
        Parameters
        ----------
        n_components : int
            Number of CSP components to extract.
        kernel : str
            SVM kernel type ('linear', 'rbf', etc.).
        """
        self.n_components = n_components
        self.kernel = kernel
        self.csp = CSP(n_components=n_components, log=True)
        self.clf = SVC(kernel=kernel)

    @property
    def model_name(self) -> str:
        return "CSP_SVM"

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train CSP + SVM. X shape: (N, C, T).
        Returns None (no training history for classical models).
        """
        # CSP expects (N, C, T)
        X_train_csp = self.csp.fit_transform(X_train, y_train)
        self.clf.fit(X_train_csp, y_train)
        return None

    def predict(self, X) -> np.ndarray:
        """Predict using trained CSP + SVM pipeline."""
        X_csp = self.csp.transform(X)
        return self.clf.predict(X_csp)

    def evaluate(self, X_test, y_test) -> float:
        """Return accuracy score."""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def fit_predict(self, X_train, y_train, X_test):
        """Convenience: train and predict in one call."""
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def save(self, dirpath: str):
        """Save CSP and SVM to pickle files."""
        from utils.io_utils import save_pickle, ensure_dir
        ensure_dir(dirpath)
        save_pickle(self.csp, os.path.join(dirpath, "csp.pkl"))
        save_pickle(self.clf, os.path.join(dirpath, "svm.pkl"))

    def load(self, dirpath: str):
        """Load CSP and SVM from pickle files."""
        from utils.io_utils import load_pickle
        self.csp = load_pickle(os.path.join(dirpath, "csp.pkl"))
        self.clf = load_pickle(os.path.join(dirpath, "svm.pkl"))

    def needs_channel_dim(self) -> bool:
        return False
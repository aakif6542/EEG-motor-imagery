# models/csp_svm.py

import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class CSPSVMModel:
    def __init__(self, n_components=4, kernel='linear'):
        """
        CSP + SVM pipeline
        """
        self.csp = CSP(n_components=n_components, log=True)
        self.clf = SVC(kernel=kernel)

    def fit(self, X_train, y_train):
        """
        Train CSP + SVM
        X shape: (N, C, T)
        """

        # CSP expects (N, C, T)
        X_train_csp = self.csp.fit_transform(X_train, y_train)

        # Train SVM
        self.clf.fit(X_train_csp, y_train)

    def predict(self, X_test):
        """
        Predict using trained model
        """
        X_test_csp = self.csp.transform(X_test)
        return self.clf.predict(X_test_csp)

    def evaluate(self, X_test, y_test):
        """
        Evaluate accuracy
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc, y_pred

    def fit_predict(self, X_train, y_train, X_test):
        """
        Convenience method
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)
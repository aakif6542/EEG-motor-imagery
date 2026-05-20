# models/cnn.py
# ============================================================
# Generic CNN for EEG motor imagery classification.
#
# PRESERVED: Original 3-block architecture with Conv2D,
#   BatchNorm, ReLU, MaxPool, GlobalAvgPool, Dense.
# IMPROVED: Inherits BaseModel, dynamic input shape,
#   save/load support.
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
)

from models.base_model import BaseModel


class CNNModel(BaseModel):
    """
    Generic 3-block CNN for EEG classification.

    Architecture (preserved from original):
        Block 1: Conv2D(16, 3×15) → BN → ReLU → MaxPool(1×4)
        Block 2: Conv2D(32, 1×15) → BN → ReLU → MaxPool(1×4)
        Block 3: Conv2D(64, 1×10) → BN → ReLU → MaxPool(1×4)
        Head:    GlobalAvgPool → Dense(64) → Dropout → Dense(1, sigmoid)
    """

    def __init__(self, n_channels=None, n_timepoints=None,
                 input_shape=None, dropout_rate=0.5,
                 learning_rate=0.001):
        """
        Parameters
        ----------
        n_channels : int
            Number of EEG channels.
        n_timepoints : int
            Number of time samples per trial.
        input_shape : tuple, optional
            Direct input shape (C, T, 1). Overrides n_channels/n_timepoints.
        dropout_rate : float
            Dropout probability.
        learning_rate : float
            Adam optimizer learning rate.
        """
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = (n_channels, n_timepoints, 1)

        self.model = self._build_model(self.input_shape)

    @property
    def model_name(self) -> str:
        return "CNN"

    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # Block 1
        x = Conv2D(16, (3, 15), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # Block 2
        x = Conv2D(32, (1, 15), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # Block 3
        x = Conv2D(64, (1, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # Classification head (GlobalAvgPool instead of Flatten)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=40, batch_size=32, **kwargs):
        """Train CNN. Returns keras History object."""
        val_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict(self, X) -> np.ndarray:
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

    def evaluate(self, X_test, y_test) -> float:
        _, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return acc

    def summary(self):
        return self.model.summary()

    def save(self, dirpath: str):
        from utils.io_utils import ensure_dir
        ensure_dir(dirpath)
        self.model.save(os.path.join(dirpath, "cnn_model.keras"))

    def load(self, dirpath: str):
        self.model = tf.keras.models.load_model(
            os.path.join(dirpath, "cnn_model.keras"))

    def needs_channel_dim(self) -> bool:
        return True
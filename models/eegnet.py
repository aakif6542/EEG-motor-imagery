# models/eegnet.py
# ============================================================
# EEGNet: A compact CNN designed specifically for EEG signals.
#
# PRESERVED: Original architecture (temporal conv → depthwise
#   spatial conv → separable conv → classification).
# IMPROVED: Inherits BaseModel, dynamic input shape,
#   save/load support.
#
# Reference: Lawhern et al., 2018
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D,
    Dropout, Flatten, Dense
)

from models.base_model import BaseModel


class EEGNet(BaseModel):
    """
    EEGNet: Compact Convolutional Neural Network for EEG.

    Architecture (preserved from original):
        1. Temporal convolution: Conv2D(F1, 1×kernel_length)
        2. Spatial depthwise: DepthwiseConv2D(C×1, depth=D)
        3. Separable convolution: SeparableConv2D(F2, 1×16)
        4. Classification: Flatten → Dense(1, sigmoid)
    """

    def __init__(self, n_channels=None, n_timepoints=None,
                 input_shape=None, dropout_rate=0.5,
                 kernel_length=64, F1=8, D=2, F2=16,
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
        kernel_length : int
            Length of temporal convolution kernel.
        F1 : int
            Number of temporal filters.
        D : int
            Depth multiplier for depthwise convolution.
        F2 : int
            Number of pointwise filters.
        learning_rate : float
            Adam optimizer learning rate.
        """
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.learning_rate = learning_rate

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = (n_channels, n_timepoints, 1)

        self.model = self._build_model(self.input_shape)

    @property
    def model_name(self) -> str:
        return "EEGNet"

    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # 1. TEMPORAL CONVOLUTION
        x = Conv2D(self.F1, (1, self.kernel_length),
                    padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)

        # 2. DEPTHWISE SPATIAL CONVOLUTION
        x = DepthwiseConv2D(
            (input_shape[0], 1),
            use_bias=False,
            depth_multiplier=self.D
        )(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(self.dropout_rate)(x)

        # 3. SEPARABLE CONVOLUTION
        x = SeparableConv2D(self.F2, (1, 16),
                            use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 8))(x)
        x = Dropout(self.dropout_rate)(x)

        # 4. CLASSIFICATION
        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=40, batch_size=32, **kwargs):
        """Train EEGNet. Returns keras History object."""
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
        self.model.save(os.path.join(dirpath, "eegnet_model.keras"))

    def load(self, dirpath: str):
        self.model = tf.keras.models.load_model(
            os.path.join(dirpath, "eegnet_model.keras"))

    def needs_channel_dim(self) -> bool:
        return True
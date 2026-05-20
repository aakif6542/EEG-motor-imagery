# models/eeg_conformer.py
# ============================================================
# EEG Conformer: A lightweight transformer-based architecture
# for EEG motor imagery classification.
#
# Architecture:
#   1. Temporal Convolution Frontend — extracts local temporal
#      features from raw EEG using 1D-style Conv2D blocks.
#   2. Transformer Encoder — applies multi-head self-attention
#      over the temporal sequence to capture long-range
#      dependencies.
#   3. Classification Head — global average pooling + dense
#      layers for binary classification.
#
# Designed to be lightweight enough to train on BNCI2014001
# and Cho2017 without excessive GPU requirements.
#
# Reference: Song et al., "EEG Conformer: Convolutional
# Transformer for EEG Decoding and Visualization", 2023
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    AveragePooling2D, Dropout, Dense, Flatten,
    LayerNormalization, GlobalAveragePooling1D,
    Reshape, Add, MultiHeadAttention
)

from models.base_model import BaseModel


# ---------------------------------------------------------------
# POSITIONAL ENCODING (Sinusoidal)
# ---------------------------------------------------------------
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal positional encoding for transformer input.

    Adds position-dependent signals to the input embeddings so
    the transformer can reason about temporal ordering.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, seq_len, d_model)
        seq_len = input_shape[1]
        d_model = input_shape[2]

        # Precompute positional encoding matrix
        positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]       # (1, d_model)

        # Compute angles
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)

        # Apply sin to even indices, cos to odd indices
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])

        # Store as non-trainable weight (batch dim added)
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, seq_len, d_model),
            initializer=tf.keras.initializers.Constant(pe[np.newaxis, :, :]),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pe

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------
# TRANSFORMER ENCODER BLOCK
# ---------------------------------------------------------------
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Single transformer encoder block with:
      - Multi-Head Self-Attention
      - Feed-Forward Network (2-layer MLP)
      - Residual connections + LayerNorm (Pre-LN variant)
      - Dropout for regularization

    Uses Pre-LN (LayerNorm before attention/FFN) for more
    stable training on small EEG datasets.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Multi-Head Self-Attention
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )

        # Feed-Forward Network
        self.ffn_dense1 = Dense(self.ff_dim, activation='gelu')
        self.ffn_dense2 = Dense(self.d_model)

        # Layer Normalization (Pre-LN)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        # Pre-LN → Multi-Head Attention → Residual
        x_norm = self.ln1(x)
        attn_out = self.mha(x_norm, x_norm, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = x + attn_out  # Residual connection

        # Pre-LN → FFN → Residual
        x_norm = self.ln2(x)
        ffn_out = self.ffn_dense1(x_norm)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = x + ffn_out  # Residual connection

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


# ---------------------------------------------------------------
# EEG CONFORMER MODEL
# ---------------------------------------------------------------
class EEGConformer(BaseModel):
    """
    EEG Conformer: Lightweight Convolutional Transformer for
    EEG Motor Imagery Classification.

    Architecture overview:
        1. Temporal Conv Frontend:
           - Conv2D(F1, 1×kernel_size) — temporal feature extraction
           - BatchNorm → ELU → AveragePool → Dropout
           - Conv2D(F2, 1×kernel_size2) — deeper temporal features
           - BatchNorm → ELU → AveragePool → Dropout
        2. Spatial Reduction:
           - Conv2D(d_model, channels×1) — compress spatial dim
           - BatchNorm → ELU
        3. Reshape to sequence: (batch, seq_len, d_model)
        4. Positional Encoding (sinusoidal)
        5. N × Transformer Encoder Blocks
        6. Classification Head:
           - LayerNorm → GlobalAveragePooling1D
           - Dense(64) → Dropout → Dense(1, sigmoid)

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_timepoints : int
        Number of time samples per trial.
    input_shape : tuple, optional
        Direct input shape (C, T, 1). Overrides n_channels/n_timepoints.
    d_model : int
        Transformer embedding dimension.
    num_heads : int
        Number of attention heads.
    num_transformer_blocks : int
        Number of stacked transformer encoder blocks.
    ff_dim : int
        Feed-forward network hidden dimension.
    conv_F1 : int
        Number of filters in first temporal conv block.
    conv_F2 : int
        Number of filters in second temporal conv block.
    conv_kernel1 : int
        Kernel size for first temporal conv.
    conv_kernel2 : int
        Kernel size for second temporal conv.
    dropout_rate : float
        Dropout probability used throughout.
    learning_rate : float
        Adam optimizer learning rate.
    """

    def __init__(self, n_channels=None, n_timepoints=None,
                 input_shape=None, d_model=64, num_heads=4,
                 num_transformer_blocks=2, ff_dim=128,
                 conv_F1=16, conv_F2=32,
                 conv_kernel1=25, conv_kernel2=15,
                 dropout_rate=0.3, learning_rate=0.0005):

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        self.conv_F1 = conv_F1
        self.conv_F2 = conv_F2
        self.conv_kernel1 = conv_kernel1
        self.conv_kernel2 = conv_kernel2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = (n_channels, n_timepoints, 1)

        self.model = self._build_model(self.input_shape)

    @property
    def model_name(self) -> str:
        return "EEGConformer"

    def _build_model(self, input_shape):
        """
        Build the EEG Conformer architecture.

        Input: (batch, channels, timepoints, 1)
        """
        n_channels = input_shape[0]
        n_timepoints = input_shape[1]

        inputs = Input(shape=input_shape)

        # ==========================================================
        # 1. TEMPORAL CONVOLUTION FRONTEND
        # ==========================================================

        # Block 1: Temporal feature extraction
        x = Conv2D(self.conv_F1, (1, self.conv_kernel1),
                   padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(self.dropout_rate)(x)

        # Block 2: Deeper temporal features
        x = Conv2D(self.conv_F2, (1, self.conv_kernel2),
                   padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(self.dropout_rate)(x)

        # ==========================================================
        # 2. SPATIAL REDUCTION
        # Compress the channel (spatial) dimension into d_model
        # ==========================================================

        x = Conv2D(self.d_model, (n_channels, 1),
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        # Shape: (batch, 1, reduced_timepoints, d_model)

        # ==========================================================
        # 3. RESHAPE TO SEQUENCE FOR TRANSFORMER
        # ==========================================================

        # Remove the spatial dim (now 1) → (batch, seq_len, d_model)
        seq_len = x.shape[2]  # reduced_timepoints after pooling
        x = Reshape((seq_len, self.d_model))(x)

        # ==========================================================
        # 4. POSITIONAL ENCODING
        # ==========================================================

        x = SinusoidalPositionalEncoding()(x)

        # ==========================================================
        # 5. TRANSFORMER ENCODER BLOCKS
        # ==========================================================

        for i in range(self.num_transformer_blocks):
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            )(x)

        # ==========================================================
        # 6. CLASSIFICATION HEAD
        # ==========================================================

        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        # ==========================================================
        # COMPILE
        # ==========================================================

        model = Model(inputs, outputs, name='EEGConformer')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=40, batch_size=32, **kwargs):
        """Train EEG Conformer. Returns keras History object."""
        val_data = (X_val, y_val) if X_val is not None else None

        # Use ReduceLROnPlateau for better convergence on small
        # EEG datasets — reduces LR when validation loss plateaus
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
        ]

        # Get verbose setting from kwargs (runner passes verbose=0)
        verbose = kwargs.get('verbose', 1)

        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels (integer array)."""
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

    def evaluate(self, X_test, y_test) -> float:
        """Return accuracy score."""
        _, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return acc

    def summary(self):
        """Print model architecture summary."""
        return self.model.summary()

    def save(self, dirpath: str):
        """Save model to directory."""
        from utils.io_utils import ensure_dir
        ensure_dir(dirpath)
        self.model.save(os.path.join(dirpath, "eeg_conformer_model.keras"))

    def load(self, dirpath: str):
        """Load model from directory."""
        self.model = tf.keras.models.load_model(
            os.path.join(dirpath, "eeg_conformer_model.keras"),
            custom_objects={
                'SinusoidalPositionalEncoding': SinusoidalPositionalEncoding,
                'TransformerEncoderBlock': TransformerEncoderBlock,
            }
        )

    def needs_channel_dim(self) -> bool:
        """EEG Conformer uses 4D input (N, C, T, 1)."""
        return True

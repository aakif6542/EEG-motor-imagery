# models/eegnet.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    SeparableConv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense
)


class EEGNet:
    def __init__(
        self,
        input_shape,
        dropout_rate=0.5,
        kernel_length=64,
        F1=8,
        D=2,
        F2=16
    ):
        self.model = self.build_model(
            input_shape,
            dropout_rate,
            kernel_length,
            F1,
            D,
            F2
        )

    def build_model(
        self,
        input_shape,
        dropout_rate,
        kernel_length,
        F1,
        D,
        F2
    ):
        inputs = Input(shape=input_shape)

        # -----------------------
        # 1. TEMPORAL CONVOLUTION
        # -----------------------
        x = Conv2D(
            F1,
            (1, kernel_length),
            padding='same',
            use_bias=False
        )(inputs)

        x = BatchNormalization()(x)

        # -----------------------
        # 2. DEPTHWISE (SPATIAL)
        # -----------------------
        x = DepthwiseConv2D(
            (input_shape[0], 1),
            use_bias=False,
            depth_multiplier=D
        )(x)

        x = BatchNormalization()(x)
        x = Activation('elu')(x)

        x = AveragePooling2D((1, 4))(x)
        x = Dropout(dropout_rate)(x)

        # -----------------------
        # 3. SEPARABLE CONVOLUTION
        # -----------------------
        x = SeparableConv2D(
            F2,
            (1, 16),
            use_bias=False,
            padding='same'
        )(x)

        x = BatchNormalization()(x)
        x = Activation('elu')(x)

        x = AveragePooling2D((1, 8))(x)
        x = Dropout(dropout_rate)(x)

        # -----------------------
        # CLASSIFICATION
        # -----------------------
        x = Flatten()(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=32):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return acc

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def summary(self):
        return self.model.summary()
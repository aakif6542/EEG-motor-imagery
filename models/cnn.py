# models/cnn.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)


class CNNModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # -----------------------
        # Block 1
        # -----------------------
        x = Conv2D(16, (3, 15), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # -----------------------
        # Block 2
        # -----------------------
        x = Conv2D(32, (1, 15), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # -----------------------
        # Block 3
        # -----------------------
        x = Conv2D(64, (1, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((1, 4))(x)

        # -----------------------
        # Instead of Flatten (IMPORTANT)
        # -----------------------
        x = GlobalAveragePooling2D()(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)

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
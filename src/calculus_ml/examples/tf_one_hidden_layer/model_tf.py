import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(2,)),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # smaller learning rate
        loss='binary_crossentropy'
    )
    return model
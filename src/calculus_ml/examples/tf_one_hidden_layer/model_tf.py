import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model():
    model = Sequential([
        Dense(4, activation='relu', input_shape=(2,)),  # hidden layer with more neurons
        Dense(1, activation='sigmoid')                  # output layer
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # smaller learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
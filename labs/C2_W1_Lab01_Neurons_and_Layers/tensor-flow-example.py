import numpy as np
import tensorflow as tf

#from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
#from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
#from tensorflow.keras.activations import sigmoid
#from lab_utils_common import dlc
#from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic

import logging


def main():
    #tensor_linear_prediction()
    logistic_regression()


def tensor_linear_prediction():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)  # (size in 1000 square feet)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)  # (price in 1000s of dollars)
    linear_layer = tf.keras.layers.Dense(units=1, activation='linear', )
    linear_layer.get_weights()
    a1 = linear_layer(X_train[0].reshape(1, 1))
    print(a1)
    w, b = linear_layer.get_weights()
    print(f"w = {w}, b={b}")
    # set_weights takes a list of numpy arrays
    set_w = np.array([[200]])
    set_b = np.array([100])
    linear_layer.set_weights([set_w, set_b])
    print(linear_layer.get_weights())
    a1 = linear_layer(X_train[0].reshape(1, 1))
    print(a1)
    alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
    print(alin)
    prediction_tf = linear_layer(X_train)
    print(prediction_tf)

def logistic_regression():
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

    model = Sequential(
        [
            tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
        ]
    )
    model.summary()

    logistic_layer = model.get_layer('L1')
    w,b = logistic_layer.get_weights()
    print(w,b)
    print(w.shape,b.shape)

    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    # set_weights takes a list of numpy arrays
    logistic_layer.set_weights([set_w, set_b])
    print(logistic_layer.get_weights())

    a1 = model.predict(X_train[0].reshape(1,1))
    print(a1)
    alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
    print(alog)

if __name__ == "__main__":
    main()

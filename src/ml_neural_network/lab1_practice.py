import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

def neuron_with_linear_regression():
    # training data
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

    # reshape the array per given rows and columns, 
    # for example, reshape(2, 3) will reshape the array to 2 rows and 3 columns
    single_training_data = X_train[0].reshape(1, 1)
    print(f"training data = {single_training_data}")

    # create one neural network layer with one neuron that will output linear regression prediction or activation
    # as the weight is not set, a random weight will be generated thus the activation
    linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
    weight = linear_layer.get_weights()
    print(f"weight = {weight}")
    
    # output = activation(dot(input, kernel) + bias)
    activation1 = linear_layer(single_training_data)
    print(f"activation = {activation1}")

    # check new weight and bias
    weight, bias = linear_layer.get_weights()
    print(f"weight = {weight}, bias = {bias}")

    # set new weight and bias
    new_weight = np.array([[200]])
    new_bias = np.array([100])
    linear_layer.set_weights([new_weight, new_bias])
    print(f"weight = {linear_layer.get_weights()}")

    # output from tensorflow
    activation2 = linear_layer(single_training_data)
    print(f"tensorflow_activation = {activation2}")

    # output from numpy
    numpy_output = np.dot(new_weight, single_training_data) + new_bias
    print(f"numpy_output = {numpy_output}")


def run():
    neuron_with_linear_regression()


if __name__ == "__main__":
    run()
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


def neuron_with_logistic_regression():
    # training data
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix

    # the Sequential model is a convenient means of creating multi-layer neural networks
    model = Sequential(
        [
            tf.keras.layers.Dense(units=1, input_dim=1,  activation = 'sigmoid', name='L1')
        ]
    )

    # show the the layers and the number of parameters in the model
    model.summary()

    # without a weight set a random weight will be generated
    logistic_layer = model.get_layer('L1')
    weights, bias = logistic_layer.get_weights()
    print(f"weight = {weights}, bias = {bias}")
    print(weights.shape, bias.shape)

    # set new weight and bias
    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    logistic_layer.set_weights([set_w, set_b])
    print(f"weight = {logistic_layer.get_weights()}")

    # reshaped training data
    reshaped_training_data = X_train[0].reshape(1, 1)
    print(f"training data = {reshaped_training_data}")

    # compute tensorflow activation
    tensorflow_activation = model.predict(reshaped_training_data)
    print(f"tensorflow_activation = {tensorflow_activation}")

    # compute numpy prediction
    numpy_prediction = sigmoidnp(np.dot(set_w,X_train[0].reshape(1, 1)) + set_b)
    print(f"numpy_prediction = {numpy_prediction}")


def sigmoidnp(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip(z, -500, 500)           # protect against overflow
    g = 1.0 / ( 1.0 + np.exp(-z) )

    return g

def run():
    print("=================================================================")
    print("==================== Neuron with Linear Regression ==============")
    print("=================================================================")
    neuron_with_linear_regression()

    print("=====================================================================")
    print("==================== Neuron with Logistic Regression ================")
    print("=====================================================================")
    neuron_with_logistic_regression()


if __name__ == "__main__":
    run()
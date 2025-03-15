import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load("data/w1.npy")
    b1 = np.load("data/b1.npy")
    w2 = np.load("data/w2.npy")
    b2 = np.load("data/b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    ### START CODE HERE ### 
    z = np.matmul(A_in, W) + b
    A_out = g(z)            
    
    ### END CODE HERE ### 
    return(A_out)

def run():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    # load dataset
    # these are images of hand written digits in 20x20 pixel format
    X, y = load_data()

    model = Sequential(
        [               
            tf.keras.Input(shape=(400,)),    #specify input size
            Dense(units=25, activation='sigmoid', name = 'layer1'),
            Dense(units=15, activation='sigmoid', name = 'layer2'),
            Dense(units=1, activation='sigmoid', name = 'layer3')
        ], name = "my_model" 
    )                            
    model.summary()

    # The parameter counts shown in the summary correspond to the number of elements in the weight and bias arrays as shown below.
    L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
    L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
    L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )

    #### Examine Weights shapes
    [layer1, layer2, layer3] = model.layers
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    # cost function and gradient descent
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )
    model.fit(
        X,y,
        epochs=20
    )

    prediction = model.predict(X[0].reshape(1,400))  # a zero
    prediction = model.predict(X[500].reshape(1,400))  # a one
    print(f" predicting a one:  {prediction}")

    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")


if __name__ == "__main__":
    run()
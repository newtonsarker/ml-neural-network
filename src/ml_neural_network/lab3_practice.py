import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))

def sigmoid(z):
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
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = sigmoid(z)               
    return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

def run():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    # load training data
    X,Y = load_coffee_data();
    print(X.shape, Y.shape)

    # normalize features
    print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

    # Define the activation function
    g = sigmoid

    # weight and bias
    W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    b1_tmp = np.array( [-9.82, -9.28,  0.96] )
    W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
    b2_tmp = np.array( [15.41] )

    X_tst = np.array([
        [200,13.9],  # postive example
        [200,17]])   # negative example
    X_tstn = norm_l(X_tst)  # remember to normalize
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

    # convert probabilities to decision
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    # easy way to predict
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")



if __name__ == "__main__":
    run()

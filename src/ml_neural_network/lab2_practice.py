import numpy as np
import matplotlib.pyplot as plt
import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

def run():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    # load training data
    X,Y = load_coffee_data();

    # normalize features
    print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

    # Tile/copy our data to increase the training set size and reduce the number of training epochs.
    Xt = np.tile(Xn,(1000,1))
    Yt= np.tile(Y,(1000,1))   
    print(Xt.shape, Yt.shape)   

    # tensorflow model
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='sigmoid', name = 'layer1'),
            Dense(1, activation='sigmoid', name = 'layer2')
        ]
    )
    model.summary()

    # the parameter counts shown in the summary correspond to the number of elements in the weight and bias arrays as shown below.
    L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
    L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

    # in the first layer with 3 units, we expect W to have a size of (2,3) and  𝑏 should have 3 elements.
    # in the second layer with 1 unit, we expect W to have a size of (3,1) and  𝑏 should have 1 element.
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

    # The model.compile statement defines a loss function and specifies a compile optimization.
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    # The model.fit statement runs gradient descent and fits the weights to the data.
    model.fit(
        Xt,Yt,            
        epochs=10,
    )

    # weight after fitting the model
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1:\n", W1, "\nb1:", b1)
    print("W2:\n", W2, "\nb2:", b2)

    # prediction
    X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    print("predictions = \n", predictions)

    # To convert the probabilities to a decision, we apply a threshold:
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    # smart prediction calculation
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")


if __name__ == "__main__":
    run()

from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def run():
    centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

    model = Sequential(
        [ 
            Dense(25, activation = 'relu'),
            Dense(15, activation = 'relu'),
            Dense(4, activation = 'softmax')    # < softmax activation here
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )
    model.fit(
        X_train,y_train,
        epochs=10
    )

    p_nonpreferred = model.predict(X_train)
    print(p_nonpreferred [:2])

    # from_logits = True
    # logits: 
    #   Logits are the raw, unnormalized outputs of the final layer of a neural network. 
    #   They are typically real-valued numbers that can be positive or negative. 
    #   To convert logits into probabilities, the softmax function is applied, 
    #   which normalizes the outputs so that they sum to 1 and can be interpreted as probabilities.
    # Why use from_logits=True?
    #   1. TensorFlow automatically applies the softmax function to the logits inside the loss function during the loss calculation.
    #   2. This avoids numerical instability that can occur when applying the softmax function separately in the model and then passing the probabilities to the loss function.
    preferred_model = Sequential(
        [ 
            Dense(25, activation = 'relu'),
            Dense(15, activation = 'relu'),
            Dense(4, activation = 'linear')   #<-- Note
        ]
    )
    preferred_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
        optimizer=tf.keras.optimizers.Adam(0.001),
    )
    preferred_model.fit(
        X_train,y_train,
        epochs=10
    )
    p_preferred = preferred_model.predict(X_train)
    print(f"two example output vectors:\n {p_preferred[:2]}")

    sm_preferred = tf.nn.softmax(p_preferred).numpy()
    print(f"two example output vectors:\n {sm_preferred[:2]}")

if __name__ == "__main__":
    run()
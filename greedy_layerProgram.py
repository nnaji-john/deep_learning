"""
Greedy layer-wise unsupervised pretraining protocol (stacked autoencoders)
Based on: Goodfellow, Bengio, & Courville (2016), Sec. 15.1 (pp. 519–522)

How to run (example):
  python greedy_pretraining.py

This script:
1) Creates a training set X (here: synthetic data for demonstration).
   Replace X with your real training set (NumPy array).
2) Trains one autoencoder layer at a time (greedy).
3) After each layer, transforms X -> H (encoded representation) for next layer.
"""

import numpy as np
import tensorflow as tf


def build_autoencoder(input_dim: int, hidden_dim: int) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    Builds a 1-hidden-layer autoencoder:
      encoder: x -> h
      decoder: h -> x_hat
      autoencoder: x -> x_hat
    """
    inp = tf.keras.Input(shape=(input_dim,), name="x")
    h = tf.keras.layers.Dense(hidden_dim, activation="relu", name="h")(inp)
    out = tf.keras.layers.Dense(input_dim, activation="linear", name="x_hat")(h)

    autoencoder = tf.keras.Model(inputs=inp, outputs=out, name=f"AE_{input_dim}_to_{hidden_dim}")
    encoder = tf.keras.Model(inputs=inp, outputs=h, name=f"ENC_{input_dim}_to_{hidden_dim}")

    h_inp = tf.keras.Input(shape=(hidden_dim,), name="h_in")
    x_hat = autoencoder.get_layer("x_hat")(h_inp)
    decoder = tf.keras.Model(inputs=h_inp, outputs=x_hat, name=f"DEC_{hidden_dim}_to_{input_dim}")

    return autoencoder, encoder, decoder


def train_one_layer_autoencoder(
    X: np.ndarray,
    hidden_dim: int,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    Trains a single autoencoder layer to reconstruct X.
    Returns (autoencoder, encoder, decoder).
    """
    input_dim = X.shape[1]
    autoencoder, encoder, decoder = build_autoencoder(input_dim, hidden_dim)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2,
    )
    return autoencoder, encoder, decoder


def greedy_layerwise_unsupervised_pretraining(
    X_raw: np.ndarray,
    layer_dims: list[int],
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Greedy layer-wise unsupervised pretraining protocol:

    Inputs:
      X_raw: raw training data (n_samples, n_features)
      layer_dims: list of hidden sizes [d1, d2, ..., dL]

    Returns:
      A dict with:
        encoders: list of pretrained encoders
        decoders: list of pretrained decoders
        autoencoders: list of pretrained autoencoders
        representations: list of X, H1, H2, ..., HL
    """
    X = X_raw.astype(np.float32)

    encoders: list[tf.keras.Model] = []
    decoders: list[tf.keras.Model] = []
    autoencoders: list[tf.keras.Model] = []
    reps: list[np.ndarray] = [X]  # store representations per layer

    for ell, hidden_dim in enumerate(layer_dims, start=1):
        print(f"Pretraining layer {ell}/{len(layer_dims)}: input_dim={X.shape[1]} -> hidden_dim={hidden_dim}")

        ae, enc, dec = train_one_layer_autoencoder(
            X=X,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Freeze the learned layer parameters (common after pretraining)
        enc.trainable = False
        dec.trainable = False
        ae.trainable = False

        autoencoders.append(ae)
        encoders.append(enc)
        decoders.append(dec)

        # Transform data for next layer: X := encoder(X)
        X = enc.predict(X, batch_size=batch_size, verbose=0)
        reps.append(X)

    return {
        "encoders": encoders,
        "decoders": decoders,
        "autoencoders": autoencoders,
        "representations": reps,
    }


if __name__ == "__main__":
    # ----------------------------
    # Example training set (replace with your real dataset)
    # ----------------------------
    np.random.seed(0)
    tf.random.set_seed(0)

    # Example: 10,000 samples, 50 features (raw input data)
    X_train = np.random.normal(size=(10000, 50)).astype(np.float32)

    # Desired hidden layer sizes for greedy pretraining (L layers)
    layer_dims = [32, 16, 8]

    results = greedy_layerwise_unsupervised_pretraining(
        X_raw=X_train,
        layer_dims=layer_dims,
        epochs=10,          # change if needed
        batch_size=256,     # change if needed
        learning_rate=1e-3  # change if needed
    )

    print("\nLearned representations:")
    for i, rep in enumerate(results["representations"]):
        name = "X_raw" if i == 0 else f"H_{i}"
        print(f"  {name}: shape={rep.shape}")

    print("\nDone. (You can now use the pretrained encoders to initialize a deep model.)")

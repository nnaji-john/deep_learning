# Johnpaul Nnaji
# University of The Cumberlands
# Deep Learning with TensorFlow 2 and Keras
# Greedy Layer-Wise Unsupervised Pretraining (protocol)
# Based on: Goodfellow, Bengio, & Courville (2016), Sec. 15.1





# Inputs:
#   X_raw: training set (raw inputs), shape (n_samples, n_features)
#   layer_dims: list like [d1, d2, ..., dL] for L hidden layers
#   AE_train(): routine that trains a single autoencoder on data and returns (encoder, decoder)
# Outputs:
#   encoders: list of trained encoder modules for each layer
#   decoders: list of trained decoder modules (optional, for reconstruction / stacking)
#   X_L: final top-level representation after L encoders

def greedy_layerwise_unsupervised_pretraining(X_raw, layer_dims):
    # Initialize storage for layer modules
    encoders = []
    decoders = []

    # The “current data” starts as the raw inputs
    X = X_raw

    # For each layer ℓ = 1..L, train one layer at a time (greedy)
    for ell, hidden_dim in enumerate(layer_dims, start=1):

        # 1) Define a 1-hidden-layer autoencoder for this stage:
        #    encoder: maps X -> H_ell
        #    decoder: maps H_ell -> X_hat
        #    objective: minimize reconstruction loss between X and X_hat
        encoder, decoder = AE_train(
            X_train=X,
            hidden_units=hidden_dim,
            loss="reconstruction"  # e.g., MSE for continuous inputs, CE for Bernoulli inputs
        )

        # 2) Freeze and store the learned parameters for this layer
        encoders.append(encoder)
        decoders.append(decoder)

        # 3) Transform data through the newly learned encoder to create features
        #    for the next layer’s unsupervised training
        X = encoder.forward(X)   # H_ell becomes the “data” for the next layer

    # Return the stack of pretrained encoders (and decoders if desired) plus final representation
    return encoders, decoders, X
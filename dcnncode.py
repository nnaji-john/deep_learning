# Johnpaul Nnaji
# University of The Cumberlands
# Deep Learning with TensorFlow 2 and Keras
# Writing recogniton with tensorflow


# pyright: reportMissingImports=false
# (Editor-only) Suppresses Pylance/Pyright missing-import warnings in VS Code.
# Does NOT affect runtime execution.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Training configuration
# ----------------------------
EPOCHS = 20                 # Number of full passes through the training set
BATCH_SIZE = 128            # Number of images per gradient update
VERBOSE = 1                 # Training output verbosity (1 = progress bar)
OPTIMIZER = tf.keras.optimizers.Adam()  # Optimization algorithm
VALIDATION_SPLIT = 0.95     # Fraction of training data used for validation (very large)

# ----------------------------
# Input / output configuration
# ----------------------------
IMG_ROWS, IMG_COLS = 28, 28                 # MNIST image size
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)       # Add channel dimension for CNN input
NB_CLASSES = 10                              # Digits 0–9 (10 classes)

# ----------------------------
# Define the CNN (LeNet-style DCNN)
# ----------------------------
def build(input_shape, classes):
    # Sequential model stacks layers in order
    model = models.Sequential()

    # Block 1: Conv -> ReLU -> MaxPool
    # Learns low-level features (edges/strokes) from the input image
    model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2: Conv -> ReLU -> MaxPool
    # Learns higher-level patterns based on features from block 1
    model.add(layers.Convolution2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten feature maps to a vector, then classify with dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))      # Fully-connected representation layer
    model.add(layers.Dense(classes, activation="softmax"))  # Output: class probabilities (0–9)

    return model

# ----------------------------
# Load MNIST dataset (train/test split provided by Keras)
# ----------------------------
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Reshape to include channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# Normalize pixel values from [0, 255] to [0, 1] for stable training
X_train, X_test = X_train / 255.0, X_test / 255.0

# Ensure correct numeric type for TensorFlow operations
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# One-hot encode labels because the model uses categorical_crossentropy
# Example: 3 -> [0,0,0,1,0,0,0,0,0,0]
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# ----------------------------
# Build and compile the model
# ----------------------------
model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

# categorical_crossentropy expects one-hot labels (which you created above)
model.compile(
    loss="categorical_crossentropy",
    optimizer=OPTIMIZER,
    metrics=["accuracy"]
)

# Print model architecture (layers and parameter counts)
model.summary()

# ----------------------------
# TensorBoard logging callback (logs saved under ./logs)
# ----------------------------
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# ----------------------------
# Train the model for 20 epochs
# validation_split takes a portion of X_train/y_train for validation monitoring
# ----------------------------
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks
)

# ----------------------------
# Evaluate on the independent test set
# ----------------------------
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])        # Test loss
print("Test accuracy:", score[1])       # Test accuracy

# ----------------------------
# Show some MNIST test images + predictions
# ----------------------------
# Randomly sample a few test images to visually inspect performance
n = 12
idx = np.random.choice(X_test.shape[0], n, replace=False)

X_sample = X_test[idx]                          # Selected images
y_true = np.argmax(y_test[idx], axis=1)         # Convert one-hot labels back to digit IDs

# Predict probabilities and choose the most likely class
y_prob = model.predict(X_sample, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Plot images with True (T) and Predicted (P) labels
plt.figure(figsize=(10, 4))
for i in range(n):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_sample[i].squeeze(), cmap="gray")
    plt.title(f"T:{y_true[i]}  P:{y_pred[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# Submission summary: accuracy after EPOCHS
# ----------------------------
test_loss, test_acc = score[0], score[1]
print(f"\nSubmission summary -> Test accuracy after {EPOCHS} epochs: {test_acc:.4f}")

'''
Summarry of results:
After training the LeNet-style DCNN on MNIST for 20 epochs, 
the model achieved a test accuracy of 97.35% with a test loss of 0.09497, 
indicating strong generalization performance on unseen handwritten digit images.


Reference :
Gulli, A., Kapoor, A., & Pal, S. (2019). Deep learning with TensorFlow 2 and Keras: Regression, ConvNets, GANs, 
RNNs, NLP, and more with TensorFlow 2 and the Keras API (2nd ed.). Packt Publishing.

'''
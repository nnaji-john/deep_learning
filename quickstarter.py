# Johnpaul Nnaji
# University of The Cumberlands
# Deep Learning
# Quickstarter Script


# import libaries
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Step 1: Load a prebuit daset from TensorFlow
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Build a neural network model that classifies images
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Step 3 train the neural network
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Step 4: Evaluate accuracy
model.evaluate(x_test,  y_test, verbose=2)

# Output: Probability_model(x_test[:5])
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
print("Predicted probabilities for the first 5 test images:")
print(probability_model(x_test[:5]))
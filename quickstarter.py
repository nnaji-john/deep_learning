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
'''
tf.Tensor(
[[1.3799891e-07 2.9123616e-08 3.6934987e-06 3.8085978e-05 1.6658526e-10
  6.7369785e-07 2.0429061e-11 9.9995470e-01 3.6606218e-07 2.2252343e-06]
 [1.1785759e-08 3.0205158e-05 9.9995852e-01 8.6074342e-06 1.2590425e-14
  2.0280843e-06 1.8723847e-08 9.8349148e-15 5.9404266e-07 1.6389736e-12]
 [2.9491872e-07 9.9840742e-01 3.0385856e-05 6.9829366e-06 4.4911671e-05
  7.1798036e-06 2.7758157e-05 1.3532924e-03 1.1990329e-04 1.9338763e-06]
 [9.9996829e-01 8.7626745e-10 2.9380148e-05 7.8861376e-11 4.4950608e-09
  5.6921725e-09 1.3257321e-06 2.4368205e-08 1.6786702e-08 8.8652945e-07]
 [1.0163906e-06 1.0523414e-07 1.2194668e-05 5.8779225e-08 9.9460876e-01
  1.9824211e-06 9.7825989e-07 5.3729971e-05 2.6380476e-07 5.3208284e-03]], shape=(5, 10), dtype=float32)

'''
# Johnpaul Nnaji
# University of The Cumberlands
# Deep Learning with TensorFlow 2 and Keras
# Auto MPG Dataset




# Loading required libraries
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Enable interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers # pyright: ignore[reportMissingImports]

print(tf.__version__)

# 1:
# Loading the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', 
                          comment='\t',
                          sep=' ', 
                          skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# Cleaning the data
# The dataset contains some missing values, which we will remove.
dataset.isna().sum()

# We would drop those rows with missing values.
dataset = dataset.dropna()

# check misssing values again
dataset.isna().sum()

# Inspect the data
sns.pairplot(dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde") # type: ignore
plt.show()

dataset.describe().transpose()


# 2:
# Separate target variable
# Split training and test data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

# Remove the target variable from the features
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalize the data
normalizer = tf.keras.layers.Normalization(axis=-1)

# Fit the state of the preprocessor to the training data
normalizer.adapt(np.array(train_features))

# print the mean and variance of each feature 
print(normalizer.mean.numpy())
print(normalizer.variance.numpy())

# Normalize the training data and print the first example
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


#3:
# Build a linear regression model between milage and horsepower
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras sequential
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
horsepower_model.summary()

# Run the untrained model to see the output
horsepower_model.predict(horsepower[:10])

# Configure the model for training
horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# Visualize the model's training progress using the stats stored in the history object.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

# 4:
def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()
plot_horsepower(x, y)

print(type(history))
print(history.history.keys())


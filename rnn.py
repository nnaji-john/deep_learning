# Johnpaul Nnaji 
# University of The Cumberlands 
# Deep Learning with TensorFlow 2 and Keras 
# Recurrent Neural Networks (RNNs): One-to-Many: learning to generate text (char-level)
# Chapter 8 style workflow (RNN text generation)



# ----------------------------
# 1) Import libraries + constants
# ----------------------------

import os
import numpy as np
import re
import shutil
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

EPOCHS = 50
BATCH_SIZE = 64
SEQ_LENGTH = 100    # input sequence length (number of characters)
BUFFER_SIZE = 10_000
EMBED_DIM = 256
RNN_UNITS = 1024
LEARNING_RATE = 1e-3

# ----------------------------
# 2) Download + prepare data
# ----------------------------
DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def download_and_load(urls):
    """Download and load text data from given URLs."""
    texts = []
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file(f"ex1_{i}.txt", url, cache_dir=".")
        text = open(p, "r", encoding="utf-8").read()
        text = text.replace("\ufeff", "")  # remove BOM if present
        text = text.replace('\n', ' ')  # remove newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
        # add it to the list
        texts.extend(text)
    return texts
texts = download_and_load([
    "https://www.gutenberg.org/cache/epub/28885/pg28885.txt",
    "https://www.gutenberg.org/files/12/12-0.txt"
])


# ----------------------------------
# 3) Create the vocabulary
# ----------------------------------
vocab = sorted(set(texts))
print("Vocab size:, {:d}".format(len(vocab)))
# Create mapping from vocab chars to ints, ints to chars
char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for c, i in char2idx.items()}

# ------------------------------------------------------------------------------------------------
# 4) use the mapping dictionaries to convert the character sequence input into an integer sequence
# ------------------------------------------------------------------------------------------------
# numericize the tests
text_as_int = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(text_as_int)

# --------------------------------------------------------
# 5) Define the network: prepare sequence for one to many
# --------------------------------------------------------
# number of character to show before asking for prediction
# sequence:[None, 100]
SEQ_LENGTH = 100
sequences = data.batch(SEQ_LENGTH + 1, drop_remainder=True)
# split input and target
def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq
sequences = sequences.map(split_train_labels)
# set up for training
# batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // SEQ_LENGTH // batch_size
dataset = sequences.shuffle(10000).batch(
    batch_size, drop_remainder=True)


# ----------------------
# 6) Define the network
# ----------------------
class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.dense(x)
        return x
vocab_size = len(vocab)
embedding_dim = 256

model = CharGenModel(vocab_size, SEQ_LENGTH, embedding_dim)
model.build(tf.TensorShape([batch_size, SEQ_LENGTH]))

# Define the loss function
def loss(label, predictions):
    return tf.keras.losses.sparse_categorical_crossentropy(
        label, 
        predictions, 
        from_logits=True
        )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss=loss)

# --------------------------------------
# 7) Run the training and evaluation loop
# --------------------------------------
def generate_text(model, prefix_string, char2idx, idx2char,
       num_chars_to_generate=1000, temperature=1.0):
   input = [char2idx[s] for s in prefix_string]
   input = tf.expand_dims(input, 0)
   text_generated = []
   model.rnn.reset_states()
   for i in range(num_chars_to_generate):
       preds = model(input)
       preds = tf.squeeze(preds, 0) / temperature
       # predict char returned by model
       pred_id = tf.random.categorical(
           preds, num_samples=1)[-1, 0].numpy()
       text_generated.append(idx2char[pred_id])
       # pass the prediction as the next input to the model
       input = tf.expand_dims([pred_id], 0)
   return prefix_string + "".join(text_generated)


# ------------------------------------
# 8) run 50 epochs of training
# ------------------------------------
num_epochs = 50
for i in range(num_epochs // 10):
    model.fit(
        dataset.repeat(), 
        epochs=10, 
        steps_per_epoch=steps_per_epoch
        #, callbacks=[checkpoint_callback, tensorboard_calback]
        )
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "ckpt_{:04d}.weights.h5".format(i*10))
    model.save_weights(checkpoint_file)
    # create generative model using the trained model so far
    gen_model = CharGenModel(vocab_size, SEQ_LENGTH, embedding_dim)
    _ = gen_model(tf.zeros([batch_size, SEQ_LENGTH], dtype=tf.int32))
    # gen_model.build(tf.TensorShape([batch_size, SEQ_LENGTH]))
    gen_model.load_weights(checkpoint_file)
    print("after epoch: {:d}".format((i+1)*10))
    print(generate_text(gen_model, "Alice ", char2idx, idx2char))
    print("---")
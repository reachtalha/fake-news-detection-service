import os
import inspect
import logging
import pickle

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

import tensorflow as tf

tf.gfile = tf.io.gfile
logging.basicConfig(level=logging.INFO)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from helpers.data_loader import DataLoader

dl = DataLoader()
dataf = dl.load_csv(os.path.join(main_dir, "data/fake-news/train.csv"))
dataf = dataf.iloc[:, [3, -1]]

dataf.columns = ["text", "label"]
dataf = dataf.sample(frac=0.1)
dataf.dropna(inplace=True)
dataf.isnull().sum()

dataf['text'] = dataf['text'].apply(str)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(dataf['text'], dataf['label'],
                                                                                    random_state=2018, test_size=0.2)

# Tokenizing the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)

X_train_seq = tokenizer.texts_to_sequences(train_inputs)
X_test_seq = tokenizer.texts_to_sequences(validation_inputs)

pickle.dump(tokenizer, open(os.path.join(main_dir, "trained_models/tokenizer_mdl.pkl"), "wb"))
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Building the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
batch_size = 32
epochs = 10

model.fit(X_train_padded, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluating the model
loss, accuracy = model.evaluate(X_test_padded, validation_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Saving the trained model
model.save(os.path.join(main_dir, "trained_models/detection_model.h5"))
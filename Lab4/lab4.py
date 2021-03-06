import keras
import numpy as np
import csv

from util import load_data, load_embeddings, bin_sentiment

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, MaxPooling1D, Flatten, Conv1D, MaxPooling2D, GRU
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 100
NUM_CLASSES = 5

"""
Create and cache matrices for all sets
If the numpy arrays have already been generated, please comment out.
"""

(raw_X_train, raw_X_test, raw_X_dev), (y_train, y_test, y_dev) = load_data()
embedding = load_embeddings()
X_train = np.zeros((len(raw_X_train), len(max(raw_X_train)), EMBEDDING_DIM))
for i, sent in enumerate(raw_X_train):
    words = sent.split()
    for j, word in enumerate(words):
        if word in embedding:
            vec = embedding[word]
            X_train[i, j] = vec
np.save('X_train.npy', X_train)
print('Saving X_train')

X_test = np.zeros((len(raw_X_test), len(max(raw_X_test)), EMBEDDING_DIM))
for i, sent in enumerate(raw_X_test):
    words = sent.split()
    for j, word in enumerate(words):
        if word in embedding:
            vec = embedding[word]
            X_test[i, j] = vec
np.save('X_test.npy', X_test)
print('Saving X_test')

X_dev = np.zeros((len(raw_X_dev), len(max(raw_X_dev)), EMBEDDING_DIM))
for i, sent in enumerate(raw_X_dev):
    words = sent.split()
    for j, word in enumerate(words):
        if word in embedding:
            vec = embedding[word]
            X_dev[i, j] = vec
np.save('X_dev.npy', X_dev)
print('Saving X_dev')

y_train = np.array(y_train).astype(np.float32)
for i in range(y_train.shape[0]):
    y_train[i] = bin_sentiment(y_train[i])
y_train = y_train.astype(np.uint8)
y_train = to_categorical(y_train, NUM_CLASSES)

y_test = np.array(y_test).astype(np.float32)
for i in range(y_test.shape[0]):
    y_test[i] = bin_sentiment(y_test[i])
y_test = y_test.astype(np.uint8)
y_test = to_categorical(y_test, NUM_CLASSES)

y_dev = np.array(y_dev).astype(np.float32)
for i in range(y_dev.shape[0]):
    y_dev[i] = bin_sentiment(y_dev[i])
y_dev = y_dev.astype(np.uint8)
y_dev = to_categorical(y_dev, NUM_CLASSES)

np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('y_dev.npy', y_dev)

"""
If you have generated the numpy arrays above: 
uncomment the section below, and comment out the section above.
"""

"""
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
x_dev = np.load('X_dev.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_dev = np.load('y_dev.npy')
"""
# Pad x test and dev data to match (100,100) shape of training data
x_test = pad_sequences(x_test, maxlen=100, padding='post', value=0.0)
x_dev = pad_sequences(x_dev, maxlen=100, padding='post', value=0.0)


def build_model(lr=None):
    model = Sequential()
    model.add(LSTM(512, input_shape=(100,100), return_sequences=True, dropout=0.4, recurrent_dropout=0.4))
    model.add(MaxPooling1D(pool_size=100))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))

    if lr is None:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rates[x]), metrics=['accuracy'])
    return model

def lr_check():
    # Test various learning rates
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    accuracies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    for x in range(len(learning_rates)):
        model = build_model(lr=learning_rates[x])

        # Train model
        model.fit(x_train, y_train, batch_size=256, epochs=10, callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=0)], validation_data=(x_dev, y_dev))

        # Calculate accuracy
        score = model.evaluate(x_test, y_test)
        accuracies[x] = score[1]

        # Print results
        print('Loss = ' + str(score[0]))
        print('Accuracy = ' + str(score[1]))

    for x in range(len(learning_rates)):
        print('Learning rate ' + str(learning_rates[x]) + ' gives accuracy ' + str(accuracies[x]))

if __name__ == '__main__':
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss')
    tensorboard = TensorBoard()
    model.fit(x_train, y_train, batch_size=256, epochs=30, callbacks=[tensorboard], validation_data=(x_dev, y_dev))

    score = model.evaluate(x_test, y_test)
    print('Accuracy = ' + str(score[1]))


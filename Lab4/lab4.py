import keras
import numpy as np
import csv

from util import load_data, load_embeddings

BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
EMBEDDING_DIM = 100

"""
Create and cache matrices for all sets
"""
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
y_train = np.digitize(y_train, BINS).astype(np.uint8)

y_test = np.array(y_test).astype(np.float32)
y_test = np.digitize(y_test, BINS).astype(np.uint8)

y_dev = np.array(y_dev).astype(np.float32)
y_dev = np.digitize(y_dev, BINS).astype(np.uint8)

np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('y_dev.npy', y_dev)
"""

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
X_dev = np.load('X_dev.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_dev = np.load('y_dev.npy')

print(X_train.shape)
print(X_test.shape)
print(X_dev.shape)

print(y_train.shape)
print(y_test.shape)
print(y_dev.shape)

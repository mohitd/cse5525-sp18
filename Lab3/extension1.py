import scipy.io.wavfile
import mfcc_utils
import os

X_train = []
y_train = []
X_test = []
y_test = []

# Read in wavfile data as (sample_rate, signal) tuples
# Digits 1-9
for i in range(1, 10):
    X_train.append(scipy.io.wavfile.read('digits/' + str(i) + 'a.wav'))
    y_train.append(str(i))

# oh and zero
X_train.append(scipy.io.wavfile.read('digits/oa.wav'))
y_train.append('o')
X_train.append(scipy.io.wavfile.read('digits/za.wav'))
y_train.append('z')

for filename in sorted(os.listdir('digit_extension')):
    X_test.append(scipy.io.wavfile.read('digit_extension/' + filename))
    y_test.append(filename[0])

acc = 0.
for i, (sr_test, signal_test) in enumerate(X_test):
    distances = []
    test_mfcc = mfcc_utils.mfcc(signal_test, sr_test)
    for sr_train, signal_train in X_train:
        train_mfcc = mfcc_utils.mfcc(signal_train, sr_train)
        distances.append(mfcc_utils.dtw(test_mfcc, train_mfcc)[0])

    min_idx = distances.index(min(distances))
    label = y_train[min_idx]
    print(label)
    print(y_test[i])
    if label == y_test[i]:
        acc += 1.

print('Accuracy: {:.4f}'.format(acc / float(len(X_test))))

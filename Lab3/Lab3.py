import scipy.io.wavfile
import mfcc_utils
import matplotlib.pyplot as plt

trainWavs = []
testWavs = []
trainMFCC = []
testMFCC = []
trainSpec = []
testSpec = []
gtLabels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'oh', 'zero']

# Read in wavfile data as (sample_rate, signal) tuples
# Digits 1-9
for i in range(1, 10):
    trainWavs.append(scipy.io.wavfile.read('digits/' + str(i) + 'a.wav'))
    testWavs.append(scipy.io.wavfile.read('digits/' + str(i) + 'b.wav'))
# Digit 'oh'
trainWavs.append(scipy.io.wavfile.read('digits/oa.wav'))
testWavs.append(scipy.io.wavfile.read('digits/ob.wav'))
# Digit zero
trainWavs.append(scipy.io.wavfile.read('digits/za.wav'))
testWavs.append(scipy.io.wavfile.read('digits/zb.wav'))

# Convert wavfile data to MFCC's
for i in range(0, len(trainWavs)):
    trainMFCC.append(mfcc_utils.mfcc(trainWavs[i][1], trainWavs[i][0]))
    testMFCC.append(mfcc_utils.mfcc(testWavs[i][1], testWavs[i][0]))

# Convert wavfile data to log spectrum
for i in range(0, len(trainWavs)):
    trainSpec.append(mfcc_utils.spectrum(trainWavs[i][1], trainWavs[i][0]))
    testSpec.append(mfcc_utils.spectrum(testWavs[i][1], testWavs[i][0]))



# Classify test examples
correctMFCC = 0
correctSpec = 0
for i in range(0, len(testMFCC)):
    testExampleMFCC = testMFCC[i]
    testExampleSpec = testSpec[i]
    gtLabel = gtLabels[i]
    minDistMFCC = mfcc_utils.dtw(testExampleMFCC, trainMFCC[0])[0]
    minDistSpec = mfcc_utils.dtw(testExampleSpec, trainSpec[0])[0]
    minIndexMFCC = 0
    minIndexSpec = 0
    for j in range(1, len(trainMFCC)):
        distMFCC = mfcc_utils.dtw(testExampleMFCC, trainMFCC[j])[0]
        distSpec = mfcc_utils.dtw(testExampleSpec, trainSpec[j])[0]
        if distMFCC < minDistMFCC:
            minDistMFCC = distMFCC
            minIndexMFCC = j
        if distSpec < minDistSpec:
            minDistSpec = distSpec
            minIndexSpec = j
    if gtLabel == gtLabels[minIndexMFCC]:
        correctMFCC += 1
    if gtLabel == gtLabels[minIndexSpec]:
        correctSpec += 1
    print ('Digit was ' + gtLabel + ' and MFCC prediction was ' + gtLabels[minIndexMFCC])
    print ('Digit was ' + gtLabel + ' and log spectrum prediction was ' + gtLabels[minIndexSpec])
print ('Accuracy with MFCC templates: ' + str(1.0 * correctMFCC / len(testMFCC)))
print ('Accuracy with log spectrum templates: ' + str(1.0 * correctSpec / len(testSpec)))

#Visualization extension
p = mfcc_utils.dtw(trainMFCC[8], trainMFCC[4])[2]
x = [i[0] for i in p]
y = [i[1] for i in p]
plt.plot(x, y)
plt.xlabel('Sample: 9a')
plt.ylabel('Sample: 5a')
plt.title('DTW 5a vs. 9a')
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
plt.show()
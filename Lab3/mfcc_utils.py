import numpy as np
import scipy.io.wavfile
import matplotlib
import matplotlib.pyplot as plt
import numpy.fft as npfft
from scipy.fftpack import dct

# constants & parameters
PRE_EMPH_COEFF = 0.95 # [0-1]
FRAME_SIZE = 0.025 # [s]
FRAME_STRIDE = 0.01 # [s]
NFFT = 512 # [power of 2]
NUM_FILTERS = 40
NUM_CEPS = 12

def preemphasis_filter(signal, pre_emph_coeff=PRE_EMPH_COEFF):
    emph_signal = signal[0]
    emph_signal = np.append(emph_signal, signal[1:] - pre_emph_coeff * signal[:-1])
    return emph_signal

def freq_to_mel(f):
    return 2595 * np.log10(1 + f / 700.)

def mel_to_freq(m):
    return 700 * (10 ** (m / 2595.) - 1)

def create_filter_bank(sample_rate, nfilters=NUM_FILTERS, nfft=NFFT):
    low_mel = 0
    # highest frequency is half the sample rate
    high_mel = freq_to_mel(sample_rate / 2.)

    # linearly space on the mel-scale
    mel_coords = np.linspace(0, high_mel, nfilters + 2)
    # convert back to frequencies
    freq_coords = mel_to_freq(mel_coords)
    # compute the bins
    bins = np.floor((nfft + 1) * freq_coords / sample_rate).astype(np.int)

    filter_bank = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilters + 1):
        prev = bins[m - 1]
        curr = bins[m]
        next = bins[m + 1]
        # apply triangular filters
        for k in range(prev, curr):
            filter_bank[m-1, k] = (k - prev) / (curr - prev)
        for k in range(curr, next):
            filter_bank[m-1, k] = (next - k) / (next - curr)
    return filter_bank

def spectrum(signal, sample_rate, nfft=NFFT):
    # frame splitting
    window_size = int(FRAME_SIZE * sample_rate)
    shift = FRAME_STRIDE * sample_rate
    starts = np.arange(0, signal.shape[-1] - window_size, shift)
    frames = np.zeros((starts.shape[-1], int(np.floor(nfft / 2 + 1))))

    hamming = np.hamming(window_size)

    for c in np.arange(0, starts.shape[-1]):
        start = int(starts[c])
        X = npfft.rfft(signal[start : start+window_size] * hamming, nfft)
        frames[c, :] = np.absolute(X) ** 2 / nfft
    return frames

def mfcc(signal, sample_rate, num_ceps=NUM_CEPS):
    # pre-emphasis
    emph_signal = preemphasis_filter(signal)
    spec = spectrum(emph_signal, sample_rate)
    filter_bank = create_filter_bank(sample_rate)
    # apply filter back to spectrum
    filtered = np.dot(spec, filter_bank.T)
    # remove zeros since we're converting to decibels
    filter_bank[filter_bank == 0] = np.finfo(float).eps
    filtered = 20 * np.log10(filtered)
    mfcc = dct(filtered)[:, 0:NUM_CEPS]
    return mfcc

def sumdist(x1,x2,distfunc):
    return np.sum(distfunc(x1, x2))

def absdiff(x, y):
    return abs(x-y)

def euclidean(x, y):
    return np.linalg.norm(x-y)

def dtw(x, y, dist=euclidean):
    points = []
    gx = 0
    gy = 0
    m = len(x)
    n = len(y)
    D = np.zeros((m, n))
    B = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            D[i, j] = min([D[i-1, j], D[i-1, j-1], D[i,j-1]]) + dist(x[i], y[j])
            idx = np.argmin([D[i-1, j], D[i-1, j-1], D[i,j-1]])
            if idx == 0:
                B[i, j, 0] = -1
                gx += 1
            elif idx == 1:
                B[i, j, 0] = -1
                B[i, j, 1] = -1
                gx += 1
                gy += 1
            else:
                B[i, j, 1] = -1
                gy += 1
            points.append([gx, gy])
    return D[-1, -1], B, points

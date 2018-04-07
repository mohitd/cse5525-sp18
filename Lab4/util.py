import numpy as np
import csv
import unicodedata

def load_data():
    """Used only once to generate all cached npy files
    """
    # load sentence_idx -> sentence
    sentence_idx_to_sentence = {}
    with open('data/datasetSentences.txt', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        # skip header row
        next(csv_reader)
        
        for line in csv_reader:
            sentence_idx_to_sentence[line[0]] = line[1]

    # load sentence -> sentiment_idx
    sentence_to_sentiment_idx = {}
    with open('data/dictionary.txt', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        # skip header row
        next(csv_reader)
        
        for line in csv_reader:
            sentence_to_sentiment_idx[line[0]] = line[1]

    # load sentiment_idx -> sentiment
    sentiment_idx_to_sentiment = {}
    with open('data/sentiment_labels.txt', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        # skip header row
        next(csv_reader)

        for line in csv_reader:
            sentiment_idx_to_sentiment[line[0]] = line[1]

    # sentence_idx -> split_label
    dataset_split = {}
    with open('data/datasetSplit.txt', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        # skip header row
        next(csv_reader)

        for line in csv_reader:
            dataset_split[line[0]] = int(line[1])

    X_train = []
    X_test = []
    X_dev = []

    y_train = []
    y_test = []
    y_dev = []

    for sentence_idx, sentence in sentence_idx_to_sentence.items():
        if not sentence in sentence_to_sentiment_idx:
            continue
        sentiment_idx = sentence_to_sentiment_idx[sentence]
        sentiment = sentiment_idx_to_sentiment[sentiment_idx]

        if dataset_split[sentence_idx] == 1:
            X_train.append(sentence)
            y_train.append(sentiment)
        elif dataset_split[sentence_idx] == 2:
            X_test.append(sentence)
            y_test.append(sentiment)
        elif dataset_split[sentence_idx] == 3:
            X_dev.append(sentence)
            y_dev.append(sentiment)

    return (X_train, X_test, X_dev), (y_train, y_test, y_dev)

def load_embeddings():
    embedding = {}
    with open('glove/glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.array(values[1:], dtype=np.float32)
            embedding[word] = vec
    return embedding

def bin_sentiment(raw_score):
    if 0.0 <= raw_score <= 0.2:
        return 0
    elif 0.2 < raw_score <= 0.4:
        return 1 
    elif 0.4 < raw_score <= 0.6:
        return 2
    elif 0.6 < raw_score <= 0.8:
        return 3
    elif 0.8 < raw_score <= 1.0:
        return 4

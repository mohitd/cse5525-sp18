from collections import Counter
import math
import pickle

# this is an example of how to parse the POS tag file and get counts
# needed for a bigram tagger 

def load_untagged_data():
    """Returns list of sentences from training data, each of which is a list of words without tags
    """

    sentences = []

    with open("pos_train.txt","r") as infile:
        for line in infile:
            
            sentence = []
            
            #
            # split line into word/tag pairs
            #
            for wordtag in line.rstrip().split(" "):
                if wordtag == "":
                    continue
                # note that you might have escaped slashes
                # 1\/2/CD means "1/2" "CD"
                # keep 1/2 as 1\/2 
                parts = wordtag.split("/")
                parts.pop()
                word = "/".join(parts)
                # add word to sentence
                sentence.append(word)
            # add sentence to sentence list
            sentences.append(sentence)
        
    return sentences

def load_training_data():
    """Returns transition and emission counts, respectively
    """
    tag_given_tag_counts = dict()
    word_given_tag_counts = dict()

    with open("pos_train.txt","r") as infile:
        for line in infile:
            #
            # first tag is the start symbol
            lasttag = "<s>"
            #
            # split line into word/tag pairs
            #
            for wordtag in line.rstrip().split(" "):
                if wordtag == "":
                    continue
                # note that you might have escaped slashes
                # 1\/2/CD means "1/2" "CD"
                # keep 1/2 as 1\/2 
                parts = wordtag.split("/")
                tag = parts.pop()
                word = "/".join(parts)
                #
                # update counters
                if tag not in word_given_tag_counts:
                    word_given_tag_counts[tag] = Counter()
                if lasttag not in tag_given_tag_counts:
                    tag_given_tag_counts[lasttag] = Counter()
                word_given_tag_counts[tag][word] += 1
                tag_given_tag_counts[lasttag][tag] += 1
                lasttag = tag

            if lasttag not in tag_given_tag_counts:
                tag_given_tag_counts[lasttag] = Counter()
            tag_given_tag_counts[lasttag]['</s>'] += 1

    return tag_given_tag_counts, word_given_tag_counts

    # examples
    """
    print ("count[NN][VB] = "+str(tag_given_tag_counts["NN"]["VB"]))
    print ("count[NN][dog] = "+str(word_given_tag_counts["NN"]["dog"]))
    """

def load_models():
    """Returns transition and emission models in log-space

    Returns:
        transition_model[t_{i-1}][t_i] = p(going to t_i from t_{i-1}) = a_{i-1, i}
        emission_model[t_i][w_i] = p(w_i|t_i)

    """
    with open('transition_model.pkl', 'rb') as f:
        transition_model = pickle.load(f)
    with open('emission_model.pkl', 'rb') as f:
        emission_model = pickle.load(f)

    return transition_model, emission_model

def load_testing_data():
    """Returns a list of the testing data where each item is a list of [(w, t)]"""
    testing_data = []

    with open("pos_test.txt","r") as infile:
        for line in infile:
            # first tag is the start symbol
            lasttag = "<s>"

            sentence = []

            # split line into word/tag pairs
            for wordtag in line.rstrip().split(" "):
                if wordtag == "":
                    continue
                # note that you might have escaped slashes
                # 1\/2/CD means "1/2" "CD"
                # keep 1/2 as 1\/2 
                parts = wordtag.split("/")
                tag = parts.pop()
                word = "/".join(parts)
                sentence.append((word, tag))
                
            testing_data.append(sentence)

    return testing_data

def accuracy(obs, pred):
    acc = 0.
    for i in range(len(obs)):
        acc += obs[i][1] == pred[i]

    return acc / len(obs)

if __name__ == '__main__':
    tag_given_tag_counts, word_given_tag_counts = load_training_data()

    words = set()
    for t1 in word_given_tag_counts:
        for wd in word_given_tag_counts[t1]:
            words.add(wd)

    tag_counts = Counter()
    for t1 in tag_given_tag_counts:
        tag_counts[t1] += sum(tag_given_tag_counts[t1][t2] for t2 in tag_given_tag_counts)

    tag_counts['</s>'] += sum(tag_given_tag_counts[t1]['</s>'] for t1 in tag_given_tag_counts)

    transition_model = dict()
    for t1 in tag_counts:
        transition_model[t1] = dict()
        for t2 in tag_counts:
            transition_model[t1][t2] = float('-inf')

    for tag_prev in tag_given_tag_counts:
        for tag_next in tag_given_tag_counts[tag_prev]:
            if tag_given_tag_counts[tag_prev][tag_next] != 0:
                transition_model[tag_prev][tag_next] = math.log(tag_given_tag_counts[tag_prev][tag_next]) - math.log(tag_counts[tag_prev])

    with open('transition_model.pkl', 'wb') as f:
        pickle.dump(transition_model, f)

    emission_model = dict()
    for t in tag_counts:
        emission_model[t] = dict()
        for wd in words:
            emission_model[t][wd] = float('-inf')

    for tag in word_given_tag_counts:
        for word in word_given_tag_counts[tag]:
            if word_given_tag_counts[tag][word] != 0:
                emission_model[tag][word] = math.log(word_given_tag_counts[tag][word]) - math.log(tag_counts[tag])

    with open('emission_model.pkl', 'wb') as f:
        pickle.dump(emission_model, f)

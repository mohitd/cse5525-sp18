from util import load_models, load_testing_data, load_untagged_data, accuracy
#from part0 import viterbi_decoding
from math import log
import random

def random_init(model):
    """Sets probabilities in model to semi-random values
    """

    randomized = {}

    for item in model:
        randomized_item = {}
        item_sum = 0.0
        for entry in model[item]:
            # start with uniform distribution
            randomized_item[entry] = 1.0 / len(item)
            # randomly vary by +- 10%
            variation = (random.random() - 0.5) / 5.0
            randomized_item[entry] += randomized_item[entry] * variation
            item_sum += randomized_item[entry]
        # normalize probabilities
        for entry in randomized_item:
            randomized_item[entry] = randomized_item[entry] / item_sum
        randomized[item] = randomized_item

    return randomized

def forward(sentences, tags, transition_model, emission_model):
    """Returns alpha matrix for all words in all sentences, given
    current transition and emission probabilities"""
    # alpha is a list of lists of dictionaries
    # alpha is a list of sentences
    #   each sentence is a list of words
    #     each word is a dictionary of tag probabilities
    alpha = []
    
    # go through each sentence separately
    for sentence in sentences:
        sentence_probs = []
        # alpha probs for 1st word in sentence is P(tag|<s>) * P(word|tag) for each tag
        tag_probs = {}
        for tag in tags:
            transition_prob = 1e-10
            if tag in transition_model['<s>']:
                transition_prob = transition_model['<s>'][tag]
                
            emission_prob = 1e-10
            if sentence[0] in emission_model[tag]:
                emission_prob = emission_model[tag][sentence[0]]
            tag_probs[tag] = transition_prob * emission_prob
        sentence_probs.append(tag_probs)

        # alpha probs for remaining words in sentence
        for t in range(1, len(sentence)):
            tag_probs = {}
            for tag in tags:
                tag_probs[tag] = 0.0
            prev_tag_probs = sentence_probs[t-1]
            curr_word = sentence[t]
            # go through each tag from previous word, make its "contribution"
            for prev_tag in prev_tag_probs:
                prev_alpha = prev_tag_probs[prev_tag]
                for curr_tag in tag_probs:
                    transition_prob = 1e-10
                    if curr_tag in transition_model[prev_tag]:
                        transition_prob = transition_model[prev_tag][curr_tag]
                    emission_prob = 1e-10
                    if curr_word in emission_model[curr_tag]:
                        emission_prob = emission_model[curr_tag][curr_word]
                    tag_probs[curr_tag] += prev_alpha * transition_prob * emission_prob
            sentence_probs.append(tag_probs)

        alpha.append(sentence_probs)

    return sentence_probs

# ice cream example data
ic_sentence = [['2', '3', '3', '2', '3', '2', '3', '2', '2', '3', '1', '3', '3', '1', '1', '1', '2', '1', '1', '1', '3', '1', '2', '1', '1', '1', '2', '3', '3', '2', '3', '2', '2']]
ic_tags = ['H', 'C']
ic_trans = {'<s>':{'C':0.5,'H':0.5,'<s/>':0.0},'H':{'C':0.1,'H':0.8,'<s/>':0.1},'C':{'C':0.8,'H':0.1,'<s/>':0.1}}
ic_emi = {'H':{'1':0.1,'2':0.2,'3':0.7},'C':{'1':0.7,'2':0.2,'3':0.1}}

transition_model, emission_model = load_models()
rand_transition_model = random_init(transition_model)
rand_emission_model = random_init(emission_model)
sentences = load_untagged_data()
#testing_data = load_testing_data()
tags = list(transition_model.keys())
avg_acc = 0.

##ic_alpha = forward(ic_sentence, ic_tags, ic_trans, ic_emi)
##for thingy in ic_alpha:
##    print thingy

##for example in testing_data:
##    words = list(zip(*example))[0]
##    pred = viterbi_decoding(words, tags, transition_model, emission_model)
##    avg_acc += accuracy(example, pred)
##
##print('Accuracy: {}'.format(avg_acc / len(testing_data)))

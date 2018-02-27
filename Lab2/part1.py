from util import load_models, load_testing_data, load_untagged_data, accuracy
# TODO: add this back in
# (unless it turns out I have misunderstood how python imports work)
# (in which case, I guess just copy over the vit_dec declaration from part0?)
#from part0 import viterbi_decoding
from math import log
from collections import defaultdict
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

    return alpha

def backward(sentences, tags, transition_model, emission_model):
    """Returns beta matrix for all words in all sentences, given
    current transition and emission probabilities"""
    # beta is a list of lists of dictionaries
    # beta is a list of sentences
    #   each sentence is a list of words
    #     each word is a dictionary of tag probabilities
    beta = []
    
    # go through each sentence separately
    for sentence in sentences:
        sentence_probs = []
        # beta probs for last word in sentence is just P(</s>|tag) for each tag
        tag_probs = {}
        for tag in tags:
            tag_probs[tag] = 1e-10
            if '</s>' in transition_model[tag]:
                tag_probs[tag] = transition_model[tag]['</s>']

        sentence_probs.append(tag_probs)

        # beta probs for remaining words in sentence
        for t in range(len(sentence) - 2, -1, -1):
            tag_probs = {}
            for tag in tags:
                tag_probs[tag] = 0.0
            next_tag_probs = sentence_probs[0]
            next_word = sentence[t+1]
            # go through each tag from next word, make its "contribution"
            for next_tag in next_tag_probs:
                next_beta = next_tag_probs[next_tag]
                emission_prob = 1e-10
                if next_word in emission_model[next_tag]:
                    emission_prob = emission_model[next_tag][next_word]
                for curr_tag in tag_probs:
                    transition_prob = 1e-10
                    if next_tag in transition_model[curr_tag]:
                        transition_prob = transition_model[curr_tag][next_tag]
                    tag_probs[curr_tag] += next_beta * transition_prob * emission_prob
            sentence_probs.insert(0, tag_probs)

        beta.append(sentence_probs)

    return beta

def maximize(alpha, beta, tags, sentences, emission, transition):
    """Returns updated Emission and Tranisition 
    models based on Expectation iterations"""
    
    obs = list(emission.values())[0].keys()
    prob_state_obs = {}
 
    #compute sum: alpha(s)*beta(s) and init P(->S,): COL L-Q on Eisner
    tot_prob = 0.0
    for key in tags:
        tot_prob += alpha[0][key] * beta[0][key]
        prob_state_obs[key] = {}
        for val in obs:
            prob_state_obs[key][val] = 0
        
    #compute p(->S), J and K columns in Eisner spreadsheet
    prob_state = {} 
    state_sums = {}
    state_sums = defaultdict(lambda:0,state_sums)
    for sentence in sentences:
        for i in range(len(alpha)):
            for key in tags:
                prob_state[key] = (alpha[i][key] * beta[i][key]) / tot_prob
                state_sums[key] += prob_state[key]
                prob_state_obs[key][sentence[i]] += prob_state[key]
            
    #compute sum of p(S->S'): columns R S T U in Eisner
    tran_sums = {}
    for sentence in sentences:
        for tag1 in tags:
            tran_sums[tag1] = {}
            for tag2 in tags:
                tran_sums[tag1][tag2] = 0
                for i in range(1, len(alpha)):     
                    tran_sums[tag1][tag2] += (alpha[i-1][tag1] * beta[i][tag2] 
                            * transition[tag1][tag2] * emission[tag2][sentence[i]] 
                            / tot_prob)
            
    #Update Emission mode: P(O,S) / P(S)
    for key in tags:
        for val in obs:
            emission[key][val] = prob_state_obs[key][val] / state_sums[key]
            
    #update Transition model: a[t-1](S)* b[t](S) * P(S'|S) * P(O|S) / Tot_Prob
    for tag1 in tags:
        for tag2 in list(transition.values())[0]:
            if (tag2 == '</s>'):
                last_prod = (alpha[len(alpha)-1][tag1] * beta[len(beta)-1][tag1]) / tot_prob 
                transition[tag1][tag2] = last_prod / state_sums[tag1]
            else:
                transition[tag1][tag2] = tran_sums[tag1][tag2] / state_sums[tag1]
    
    #Round about way of setting the start tag transition prob's.
    #Attempted to make it scale
    start_tag = [item for item in transition.keys() if item not in tags]
    for tag1 in start_tag:
        for tag2 in list(transition.values())[0]:
            if(tag2 not in transition.keys()):
                transition[tag1][tag2] = 0
            else:
                transition[tag1][tag2] = (alpha[0][tag2] * beta[0][tag2]) / tot_prob
              
    
# ice cream example data
# TODO: remove when finished with testing
ic_sentence = [['2', '3', '3', '2', '3', '2', '3', '2', '2', '3', '1', '3', '3', '1', '1', '1', '2', '1', '1', '1', '3', '1', '2', '1', '1', '1', '2', '3', '3', '2', '3', '2', '2']]
ic_tags = ['C', 'H']
ic_trans = {'<s>':{'C':0.5,'H':0.5,'</s>':0.0},'C':{'C':0.8,'H':0.1,'</s>':0.1},'H':{'C':0.1,'H':0.8,'</s>':0.1}}
ic_emi = {'C':{'1':0.7,'2':0.2,'3':0.1},'H':{'1':0.1,'2':0.2,'3':0.7}}

# TODO: add back in
transition_model, emission_model = load_models()
fb_transition_model = random_init(transition_model)
fb_emission_model = random_init(emission_model)
sentences = load_untagged_data()
testing_data = load_testing_data()
tags = list(transition_model.keys())
avg_acc = 0.

# TODO: remove when finished with testing
test_iter = 10
for i in range(test_iter):
    ic_alpha = forward(ic_sentence, ic_tags, ic_trans, ic_emi)[0]
    ic_beta = backward(ic_sentence, ic_tags, ic_trans, ic_emi)[0]
    maximize(ic_alpha, ic_beta, ic_tags, ic_sentence, ic_emi, ic_trans)
    
print("Emissions model: ")
for item in ic_emi.keys():
    print(item)
    print(ic_emi[item])
print("\nTransition model:")
for item in ic_trans:
    print(item)
    print(ic_trans[item])

# run the forward-backward algorithm
#num_iter = 1 # TODO: set to 10 (or whatever) (or use convergence test instead)
#for i in range (0, num_iter):
#    print("Starting alpha")
#    alpha = forward(sentences, tags, fb_transition_model, fb_emission_model)
#    print("Starting beta")
#    beta = backward(sentences, tags, fb_transition_model, fb_emission_model)
#    print("Starting max")
#    maximize(alpha, beta, tags, sentences, fb_emission_model, fb_transition_model)
    

# TODO: add back in    
# final step: same as part 0, except using models learned from fb algorithm
#for example in testing_data:
#    words = list(zip(*example))[0]
#    pred = viterbi_decoding(words, tags, fb_transition_model, fb_emission_model)
#    avg_acc += accuracy(example, pred)
#
#print('Accuracy: {}'.format(avg_acc / len(testing_data)))

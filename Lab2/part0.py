from util import load_models, load_testing_data
from math import log

def viterbi_decoding(obs, tags, transition_model, emission_model):
    viterbi = [dict()] * (len(obs) + 1)
    backpointers = [dict()] * (len(obs) + 1)

    first_wd = obs[0][0]
    for tag in tags:
        viterbi[0][tag] = transition_model['<s>'][tag] + emission_model[tag][first_wd]
        backpointers[0][tag] = 0

    for t in range(1, len(obs)):
        for tag in tags:
            viterbi[t][tag] = float('-inf')
            backpointers[t][tag] = -1
            p_emission = emission_model[tag][obs[t][0]]
            for i, tag_prime in enumerate(tags):
                v = viterbi[t-1][tag_prime] + transition_model[tag_prime][tag] + p_emission
                if v > viterbi[t][tag]:
                    viterbi[t][tag] = v
                    backpointers[t][tag] = i

    # last node
    viterbi[len(obs)] = float('-inf')
    backpointers[len(obs)] = -1
    for i, tag_prime in enumerate(tags):
        v = viterbi[len(obs)-1][tag_prime] + transition_model[tag_prime]['</s>']
        if v > viterbi[len(obs)]:
            viterbi[len(obs)] = v
            backpointers[len(obs)] = i

    backtrace = []
    idx = backpointers[len(obs)]
    for t in reversed(range(len(obs))):
        backtrace.append(tags[idx])
        idx = backpointers[t][tags[idx]]

    backtrace.reverse()
    return backtrace

transition_model, emission_model = load_models()
testing_data = load_testing_data()
tags = list(transition_model.keys())

print(viterbi_decoding(testing_data[0], tags, transition_model, emission_model))

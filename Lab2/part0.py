from util import load_models, load_testing_data, accuracy
from math import log

def viterbi_decoding(obs, tags, transition_model, emission_model):
    """Implements the Viterbi algorithm to produce part-of-speech tags

    Arguments:
        obs {list} -- list of words observation
        tags {list} -- state space of tags
        transition_model {dict[dict]} -- transition_model[prev_tag][next_tag]
        emission_model {dict[dict]} -- emission_model[tag][word]

    Returns:
        list -- list of predicted tags computed from the Viterbi algorithm
    """
    sent_len = len(obs)
    viterbi = [dict()]
    backpointers = [dict()]

    # start node
    for tag in tags:
        if obs[0] in emission_model[tag]:
            p_emission = emission_model[tag][obs[0]]
        else:
            p_emission = -1e10
        viterbi[0][tag] = transition_model['<s>'][tag] + p_emission
        backpointers[0][tag] = 0

    for t in range(1, sent_len):
        v_t = dict()
        b_t = dict()
        for curr_tag in tags:
            max_score = float('-inf')
            max_score_idx = -1
            if obs[t] in emission_model[curr_tag]:
                p_emission = emission_model[curr_tag][obs[t]]
            else:
                p_emission = -1e10

            for i, prev_tag in enumerate(tags):
                score = viterbi[-1][prev_tag] + transition_model[prev_tag][curr_tag] + p_emission

                if score > max_score:
                    max_score = score
                    max_score_idx = i

            # end for over s'
            v_t[curr_tag] = max_score
            b_t[curr_tag] = max_score_idx

        # end for over s
        viterbi.append(v_t)
        backpointers.append(b_t)

    # final node
    max_score = float('-inf')
    max_score_idx = -1
    for i, prev_tag in enumerate(tags):
        score = viterbi[-1][prev_tag] + transition_model[prev_tag]['</s>']
        if score > max_score:
            max_score = score
            max_score_idx = i
    backpointers.append(max_score_idx)

    backtrace = []
    idx = backpointers[-1]
    for t in reversed(range(sent_len)):
        backtrace.append(tags[idx])
        idx = backpointers[t][tags[idx]]

    backtrace.reverse()
    return backtrace

transition_model, emission_model = load_models()
testing_data = load_testing_data()
tags = list(transition_model.keys())
avg_acc = 0.

for example in testing_data:
    words = list(zip(*example))[0]
    pred = viterbi_decoding(words, tags, transition_model, emission_model)
    avg_acc += accuracy(example, pred)

print('Accuracy: {}'.format(avg_acc / len(testing_data)))

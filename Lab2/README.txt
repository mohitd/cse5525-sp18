CSE 5525
Lab 2

Please run 'python3 util.py' first to generate the transition and emission models from the training data

Commands:
python3 util.py -- computes transition and emission models (transition_model.pkl and emission_model.pkl, respectively)

python3 part0.py -- computes accuracy of Viterbi decoding trained on counts of training set on the testing set

python3 icecream.py -- performs the forward-backward algorithm on the ice cream data set provided in the Eisner spreadsheet

python3 part1.py -- computes the transition and emission models using the forward-backward algorithm

We included the working ice cream example from class as a proof of concept, as we ran into a few issues
upscaling the algorithm for the sentence data. The ice cream implementation works just as the example in the
Eisner spreadsheet. When we scaled up to work on the sentence data we ran into quite a few issues, most were
resolved but the performance issues remained. Currently we work with a subset of the sentences, good news
is as we increase our sample size our accuracy against the test set improves. Bad news it takes a while to get
decent results. 

We would assume a correct version of the forward-backward algorithm would outperform the viterbi decoding
given enough iterations as it has more information to work with. 
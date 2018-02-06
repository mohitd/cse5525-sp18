Group 17:
Mohit Deshpande
Benjamin Trevor
Brad Pershon


PART 1:
How to run the Poker Hand Decoder:

NOTE: to be used with test hands written in "rank-then-suit" order

0. Go into the prob1 directory

1. Compile the massive FST:
fstcompile --isymbols=cards.voc --osymbols=outcomes.voc massive.fst.txt massive.fst

2. Compose your poker hand FSA with the massive FST:
fstcompose testhand.fsa massive.fst testresult.fst

3. Determine the shortest path through the result:
fstshortestpath testresult.fst > testpath.fst

4. The output of the shortest path FST is the score of the hand

Structure of the Poker Hand Decoder:

The Poker Hand Decoder (massive.fst) is essentially one large FST that scans the hand
one card at a time until it can determine the score of the hand, at which point it
outputs the result and jumps to the final state.

At first, this may seem infeasible, as even the reduced deck of the problem (only ranks
8-Ace) present more than 11,000,000 possibilities.  However, we can use the structure of
the problem to narrow the search field significantly.

First, the fact that the ranks will come in non-decreasing order is a key factor.  This means,
for example, that once we see an Ace, we know that the rest of the hand is all Aces, as Ace
is the highest card.  This also means that hands beginning with Ace are impossible, as a deck
only contains 4 Aces.

Another important factor is that suits only matter for a small number of hands.  For any hand
with duplicate ranks (One Pair, Two Pair, Three of a Kind, Full House, Four of a Kind), the suits
of the five cards cannot possibly all be the same, so there is no reason to examine them.
Suits only need to be checked in cases where all ranks in the hand are different.

Using these principles, we can reduce the 11,000,000 possibilities to fewer than 1,000, a
much more reasonable number.

Each state in the FST (except for the initial and final states) indicates something about
what cards have been examined so far.  For example, state 8910 indicates that the first three
ranks in the hand were an Eight, a Nine, and a Ten.  States beginning with 5 are used to track
suits and are used to tell the difference between a Straight and a Straight Flush or Royal Straight
Flush, or the difference between a Flush and a scoreless hand.



PART 2:
How to compute answers:

1. Compile dur.fst
./create_dur.sh

2. Compile dict.fst
./create_dict.sh

3. Compile lm.fsa
./create_lm.sh

4. Run compose all of those together with all of the inputs to produce answers in ./answers directory
./run_model.sh

5. View answers
./view_answers.sh

dur.fst is was created using the create_seq2phn_fsts.py Python script to create an FST for each phone. Then we take the union over all of them and apply Kleene's closure. The result converts sequences of different phones into a unique phones, with at least 3 consecutive phones for separation.

dict.fst is created by writing an FST for each different digit comprised of its fundamental phones. Then we take the union over all of them and apply Kleene's closure, similar to dur.fst. the result converts sequences of phones to their digit representation.

Finally, lm.fsa is simply an FSA that accepts only words in the language, which, in our case, is digits of the English language.

To apply all of these, we simply compose dur.fst, dict.fst, lm.fsa, and the input sentence together and compute the maximum likelihood path. We store the results for each in ./answers.

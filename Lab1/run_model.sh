#!/bin/bash

for sent in prob2/sents/*.fsa; do
    base=${sent##*/}
    fstcompose $sent dur.fst output.fst
    fstcompose output.fst dict.fst output.fst
    fstcompose output.fst lm.fsa output.fst
    fstshortestpath output.fst > answers/$base
done

rm output.fst


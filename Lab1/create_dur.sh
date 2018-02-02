#!/bin/bash

# convert proto-fst's *.fst.txt to real fsts
for protofst in seq2phn_fsts/*.fst.txt; do
    base=${protofst##*/}
    fstcompile --isymbols=prob2/phn.voc --osymbols=prob2/phn.voc --keep_isymbols --keep_osymbols $protofst > seq2phn_fsts/${base%.fst.txt}.fst
done

c=0
first=0
for fst in seq2phn_fsts/*.fst; do
    if [[ c -eq 0 ]]; then
        # store the first
        first=$fst
        ((c++))
    elif [[ c -eq 1 ]]; then
        # combine the first two
        fstunion $first $fst dur.fst
        ((c++))
    else
        # fold in the rest
        fstunion $fst dur.fst dur.fst
    fi
done

fstclosure dur.fst dur.fst


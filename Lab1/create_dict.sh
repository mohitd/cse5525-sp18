#!/bin/bash

# convert proto-fst's *.fst.txt to real fsts
for protofst in phnToWord/*.txt; do
    base=${protofst##*/}
    fstcompile --isymbols=prob2/phn.voc --osymbols=prob2/words.voc --keep_isymbols --keep_osymbols $protofst > phnToWord/${base%.txt}.fst
done

c=0
first=0
for fst in phnToWord/*.fst; do
    if [[ c -eq 0 ]]; then
        # store the first
        first=$fst
        ((c++))
    elif [[ c -eq 1 ]]; then
        # combine the first two
        fstunion $first $fst dict.fst
        ((c++))
    else
        # fold in the rest
        fstunion $fst dict.fst dict.fst
    fi
done

fstclosure dict.fst dict.fst


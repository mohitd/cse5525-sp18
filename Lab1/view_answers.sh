#!/bin/bash

for answer in answers/*.fsa; do
    fstproject --project_output $answer | fstrmepsilon | fstprint --osymbols=prob2/words.voc
done


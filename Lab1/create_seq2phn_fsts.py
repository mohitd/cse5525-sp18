# CSE 5525
# Mohit Deshpande

fst_template = """0 1 {0} -
1 2 {0} - 
2 3 {0} {0}
3 3 {0} - 0.3010299957
3"""

with open('prob2/phn.voc') as f:
    lines = f.readlines()

del lines[0]

for line in lines:
    phone, _ = line.split()
    with open('seq2phn_fsts/{}.fst.txt'.format(phone), 'w') as f:
        f.write(fst_template.format(phone))

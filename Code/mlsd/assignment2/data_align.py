# deep learning for structured data
# author: vlad niculae <v.niculae@uva.nl>
# license: MIT

from os.path import join
import numpy as np


def extract_alignment(seq1, seq2):

    # code: 0=match, 1=insert seq1, 2=insert seq2.

    alignment = []

    assert len(seq1) == len(seq2)

    compact_seq1 = []
    compact_seq2 = []

    ix1 = 0
    ix2 = 0
    pairs = []

    for i in range(len(seq1)):
        assert not seq1[i] == seq2[i] == '-'
        if seq1[i] == '-':
            alignment.append(2)
            compact_seq2.append(seq2[i])
            ix2 += 1
        elif seq2[i] == '-':
            alignment.append(1)
            compact_seq1.append(seq1[i])
            ix1 += 1
        else:
            alignment.append(0)
            compact_seq1.append(seq1[i])
            compact_seq2.append(seq2[i])
            pairs.append((ix1, ix2))
            ix1 += 1
            ix2 += 1

    return "".join(compact_seq1), "".join(compact_seq2), alignment, pairs


def load():

    max_len = 100
    # try a higher max-len if you have patience, e.g.,
    # max_len = 200

    data = []
    with open(join("data", "pfam_protein_pairs.txt")) as f:
        for line in f:

            s1, s2 = line.strip().split()

            if len(s1) + len(s2) < max_len:
                data.append(extract_alignment(s1, s2))

    ix = np.random.RandomState(42).permutation(len(data))

    valid_ix = set(ix[:len(data) // 2])

    train = []
    valid = []

    for i, entry in enumerate(data):
        if i in valid_ix:
            valid.append(entry)

        else:
            train.append(entry)

    print(f"{len(train)} train, {len(valid)} valid")

    return train, valid

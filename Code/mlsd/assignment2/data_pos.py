# deep learning for structured data
# author: vlad niculae <v.niculae@uva.nl>
# license: MIT

from os.path import join

def conll_to_xy(sent):
    x, y = [], []
    for line in sent.split("\n"):
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        x.append(fields[1])
        y.append(fields[3])
    return x, y


def read_all_conll_sents(doc):
    all_x = []
    all_y = []
    for sent in doc.split("\n\n"):
        x, y = conll_to_xy(sent)
        all_x.append(x)
        all_y.append(y)
    return all_x, all_y


def load():



    with open(join("data", "pos", "nl_alpino-ud-mini-train.conllu")) as f:
        doc = f.read()
    train_x, train_y = read_all_conll_sents(doc)

    with open(join("data", "pos", "nl_alpino-ud-mini-valid.conllu")) as f:
        doc = f.read()
    valid_x, valid_y = read_all_conll_sents(doc)

    with open(join("data", "pos", "nl_alpino-ud-mini-test.conllu")) as f:
        doc = f.read()
    test_x, test_y = read_all_conll_sents(doc)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

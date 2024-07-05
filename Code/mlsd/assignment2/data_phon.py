# deep learning for structured data
# author: vlad niculae <v.niculae@uva.nl>
# license: MIT

from os.path import join
import torch

def load():
    return torch.load(join("data", "phon", "librispeech_subset.pt"))


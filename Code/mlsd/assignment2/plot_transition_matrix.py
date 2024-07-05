# deep learning for structured data
# author: vlad niculae <v.niculae@uva.nl>
# license: MIT

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    trans, tag_vocab = torch.load("learned_trans.pt")
    plt.figure()

    maxval = trans.abs().max()
    plt.imshow(trans.detach().numpy(), cmap=plt.cm.PuOr, vmin=-maxval,
               vmax=maxval)
    plt.xticks(np.arange(len(tag_vocab)), tag_vocab, rotation=90)
    plt.yticks(np.arange(len(tag_vocab)), tag_vocab)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()

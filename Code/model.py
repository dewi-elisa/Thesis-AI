import os
import re
import copy
import nltk
import configargparse

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data
import utils
import opts


class Encoder(nn.Module):
    def __init__(self, opt, word2id, id2word, device):
        super(Encoder, self).__init__()
        self.device = device
        self.word2id = word2id
        self.id2word = id2word
        self.num_tokens = len(word2id)
        self.embedding_dim = opt.embedding_dim

        self.embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=0)
        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.embedding_dim,
                               batch_first=True)
        self.linear = nn.Linear(in_features=self.embedding_dim,
                                out_features=1,
                                bias=True)

    def forward(self, tokens):
        print(tokens)

        batch_size, max_len = tokens.size()

        # Embed each token
        embedded = self.embedding(tokens)
        embedded = F.relu(F.dropout(embedded, p=0.1))

        # Score each token
        encoder_outputs, _ = self.encoder(embedded)
        scores = self.linear(encoder_outputs).view(batch_size, -1)

        print("scores:")
        print(scores)

        # What happens here? It does not seem to change anyhting
        # Does it replace the values in score with -inf when they are <= 0?
        # But there are no tokens <= 0 in the vocab 
        # (is that why it does not seem to change anything?)
        # masked_scores = scores.masked_fill(tokens.gt(0).bitwise_not(), -float('inf'))
        masked_scores = scores

        print("masked scores:")
        print(masked_scores)

        # Apply sigmoid
        prob_tokens = torch.sigmoid(masked_scores)

        # Sample a mask with Bernoulli
        mask = torch.distributions.bernoulli.Bernoulli(prob_tokens).sample().to(torch.bool)

        print("mask")
        print(mask)

        # Apply the mask
        # Does not work (need to know the indices of the 1s)
        subsentence = tokens[mask]
        print("subsentence:")
        print(subsentence)
        print([self.id2word[word.item()] for word in subsentence])
        print()

        return subsentence


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()

#     def forward(self, tokens):

#         return sentence

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="train.py")

    opts.basic_opts(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.eval_opts(parser)

    opt = parser.parse_args()
    exp = utils.name_exp(opt)
    device = utils.init_device()
    utils.init_seed(opt.seed)

    tokenizer = data.build_tokenizer(opt)
    word2id, id2word = data.build_vocab(opt, exp, tokenizer)

    # print(list(id2word.keys())[:100])

    encoder = Encoder(opt, word2id, id2word, device)

    loaders = data.build_loaders(opt, tokenizer, word2id)
    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders
    for batch_index, batch in enumerate(train_ae_loader):
        src_seqs, trg_seqs, src_lines, trg_lines = batch  # Encoded form
        print("sentence:")
        print(src_lines)
        print(trg_lines)
        src_seqs = src_seqs.to(device)
        trg_seqs = trg_seqs.to(device)
        encoder(src_seqs)

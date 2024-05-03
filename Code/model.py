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
        embedded = F.relu(F.dropout(embedded, p=0.1))  # why relu? -> try also without relu

        # Score each token
        encoder_outputs, _ = self.encoder(embedded)
        scores = self.linear(encoder_outputs).view(batch_size, -1)

        print("scores:")
        print(scores)

        # What happens here? It does not seem to change anyhting
        # Does it replace the values in score with -inf when they are <= 0?
        # But there are no tokens <= 0 in the vocab
        # (is that why it does not seem to change anything?)
        # answer: padding = 0 -> for when working in batches, 
        # in that case uncomment the next line
        # masked_scores = scores.masked_fill(tokens.gt(0).bitwise_not(), -float('inf'))
        masked_scores = scores

        print("masked scores:")
        print(masked_scores)

        # Apply sigmoid
        prob_tokens = torch.sigmoid(masked_scores)

        # Sample a mask with Bernoulli
        mask = torch.distributions.bernoulli.Bernoulli(prob_tokens).sample().to(torch.bool)

        print("mask:")
        print(mask)

        # Apply the mask
        subsentence = tokens[mask]

        print("subsentence:")
        print(subsentence)
        print([self.id2word[word.item()] for word in subsentence])
        print()

        return subsentence


class Decoder(nn.Module):
    def __init__(self, opt, word2id, id2word, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word2id = word2id
        self.id2word = id2word
        self.num_tokens = len(word2id)
        self.embedding_dim = opt.embedding_dim

        self.encoder_embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                              embedding_dim=self.embedding_dim,
                                              padding_idx=0)
        self.decoder_embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                              embedding_dim=self.embedding_dim * 2,  # noqa
                                              padding_idx=0)

        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.embedding_dim,
                               bidirectional=True,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=self.embedding_dim * 4,
                               hidden_size=self.embedding_dim * 2,
                               batch_first=True)

    def forward(self, tokens, trg_seqs):
        # in the code of the paper they build a dynamic vocab here, why is this needed?

        # embed the tokens
        embedded = self.encoder_embedding(tokens)
        embedded = F.relu(F.dropout(embedded, p=0.1))

        # encode the tokens
        # in the code of the paper encoder_hidden is split into (encoder_last_hidden, _), why?
        # as I understand from the documentation and the fact that we have a bidirectional lstm
        # this is because the first one is the last hidden state from left-to-right and the other
        # one is the last hidden state from right-to-left
        # why do we need to use the left-to-right one and not the right-to-left one?
        # left-to-right feels intuitive to me too, but I am curious as to why
        encoded, (encoder_hidden, _) = self.encoder(embedded)

        print('encoded:')
        print(encoded)

        # decode
        batch_size, max_seq_len = trg_seqs.size()

        print('batch_size:')
        print(batch_size)  # 1

        last_word = torch.tensor([self.word2id["<sos>"]] * batch_size).to(self.device)
        # the paper code uses dim=1 -> because of batches? it gives me an error with dim=1
        encoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=0)
        decoder_hidden = (encoder_hidden.view(1, batch_size, self.embedding_dim * 2),
                          torch.zeros(1, batch_size, self.embedding_dim * 2).to(self.device))

        print('last word:')
        print(last_word)  # tensor([1])

        print('encoder hidden:')
        print(encoder_hidden.size())  # torch.Size([600])

        sentence = [last_word.item()]

        for decoder_step in range(max_seq_len):
            # embed the last generated word (or the <sos> symbol if there are none)
            # the next line gives me the folowwing error:
            # TypeError: Embedding.forward() takes 2 positional arguments but 3 were given
            embedded2 = self.decoder_embedding(last_word, encoder_hidden).view(batch_size, -1)
            embedded2 = F.relu(F.dropout(embedded2, p=0.1))
            embedded3 = torch.cat((embedded2, encoder_hidden), dim=1).view(batch_size,
                                                                           1,
                                                                           self.embedding_dim * 4)

            # decode the embedded tokens -> generate the full sentence
            decoded, decoder_hidden = self.decoder(embedded3, decoder_hidden)

            print('decoded:')
            print(decoded)

            # in the code of the paper they use a global attention mechanism (how does this work again?)
            # looking at the MLSD slides, we have a context vector q
            #   -> last hidden state (of the decoder?)
            # and an embedding of the words z_i
            #   -> the output of the decoder encoder?

            # and a mask softmax based on src_lens (what is src_lens?)
            # what does this part do exactly?
            # does it take a softmax over the attention mask
            # generated in the global attention mechanism step?

            # and a copy generator (what is a copy generator?)
            # in the paper, appendix C, they say something about generating
            # the next token from a fixed vocabulary or copy one of the input tokens
            # is that about the copy generator part?

            last_word = decoded
            sentence = sentence + [last_word.item()]

        print('sentence:')
        print(sentence)
        print([self.id2word[word.item()] for word in sentence])
        return sentence

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
    decoder = Decoder(opt, word2id, id2word, device)

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
        keywords = encoder(src_seqs)
        predicted = decoder(keywords, trg_seqs)

        print(predicted)

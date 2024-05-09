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
                                              embedding_dim=self.embedding_dim * 2,
                                              padding_idx=0)

        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.embedding_dim,
                               bidirectional=True,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=self.embedding_dim * 4,
                               hidden_size=self.embedding_dim * 2,
                               batch_first=True)

        self.attn = nn.Linear(in_features=self.embedding_dim * 2,
                              out_features=self.embedding_dim * 2)

        self.W = nn.Linear(in_features=self.embedding_dim * 4,
                           out_features=self.embedding_dim * 4)

        self.linear = nn.Linear(in_features=self.embedding_dim * 4,
                                out_features=self.num_tokens,
                                bias=True)

    def forward(self, tokens, trg_seqs):
        batch_size = 1  # for now, otherwise uncomment the next line
        # batch_size = tokens.size(0)

        # build vocab
        # Q: in the code of the paper they build a dynamic vocab here, why is this needed?
        # A: not necessary for now, just use a single static vocab
        # new Q: can I just use word2id and id2word from before?

        # embed the tokens
        embedded = self.encoder_embedding(tokens)
        embedded = F.relu(F.dropout(embedded, p=0.1))

        # encode the tokens
        encoded, (encoder_hidden, _) = self.encoder(embedded)

        print('encoded:')
        print(encoded)

        # decode
        batch_size, max_seq_len = 1, tokens.size()[0]  # for now, otherwise uncomment next line
        # batch_size, max_seq_len = tokens.size()

        print('batch_size:')
        print(batch_size)  # 1

        last_word = torch.tensor([self.word2id["<sos>"]] * batch_size).to(self.device)
        # Q: the paper code uses dim=1 -> because of batches? it gives me an error with dim=1
        # A: yes. because of batches, you can also try dim=-1
        encoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=-1)
        decoder_hidden = (encoder_hidden.view(1, batch_size, self.embedding_dim * 2),
                          torch.zeros(1, batch_size, self.embedding_dim * 2).to(self.device))

        print('last word:')
        print(last_word)

        sentence = [last_word.item()]

        for decoder_step in range(max_seq_len):
            # embed the last generated word (or the <sos> symbol if there are none)
            embedded2 = self.decoder_embedding(last_word).view(batch_size, -1)
            embedded2 = F.relu(F.dropout(embedded2, p=0.1))

            # concatenate the context vector to the embedding
            embedded3 = torch.cat((embedded2, encoder_hidden.view(batch_size, -1)), dim=-1)

            # decode the embedded token -> generate the next word
            decoded, decoder_hidden = self.decoder(
                embedded3.view(batch_size, 1, self.embedding_dim * 4),
                decoder_hidden)

            print('decoded:')
            print(decoded)

            # Luong's global attention
            # mask = tokens.gt(0).unsqueeze(2).expand_as(encoded).float()
            mask = tokens.gt(0).unsqueeze(1).expand_as(encoded).float()
            encoded = encoded.mul(mask)
            # attn_prod = torch.bmm(self.attn(decoder_hidden[0].transpose(0, 1)),
            #                       encoded.transpose(1, 2))
            attn_prod = torch.bmm(self.attn(decoder_hidden[0].transpose(0, 1)),
                                  encoded.transpose(0, 1).unsqueeze(0))

            # mask softmax
            max_key_len = max_seq_len  # check this
            mask_attn = tokens.gt(0).view(
                batch_size, 1, max_key_len).to(self.device)
            attn_prod.masked_fill_(mask_attn.bitwise_not(), -float('inf'))
            attn_weights = F.softmax(attn_prod, dim=2)
            attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)
            context = torch.bmm(attn_weights, encoded.unsqueeze(0))
            hc = torch.cat(
                [decoder_hidden[0].squeeze(0), context.squeeze(1)], dim=1)
            out_hc = self.W(hc)
            decoder_output = self.linear(out_hc)
            p_word = torch.zeros([batch_size, len(self.word2id)]).to(self.device)
            p_word[:, :len(self.word2id)] = F.softmax(decoder_output, dim=1)

            # copy generator
            # skipped for now

            last_word = decoder_output.argmax().item()
            sentence = sentence + [last_word]
            print('sentence:')
            print(sentence)
            print([self.id2word[word] for word in sentence])
            last_word = torch.tensor([last_word] * batch_size).to(self.device)

        print('final sentence:')
        print(sentence)
        print([self.id2word[word] for word in sentence])
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

# in the code of the paper they also initialize the wheigts. Is this necessary?

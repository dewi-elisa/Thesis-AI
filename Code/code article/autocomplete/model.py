import os
import re
import copy
import nltk

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data


class Encoder(nn.Module):
    def __init__(self, opt, word2id, id2word, device,
                 tokenizer=None):
        super(Encoder, self).__init__()
        self.device = device
        self.word2id = word2id
        self.id2word = id2word
        self.num_tokens = len(word2id)

        # For uniform encoder
        self.uniform_encoder = opt.uniform_encoder  # If true, always uniform
        self.uniform_keep_rate = opt.uniform_keep_rate

        # For stopword encoder
        self.stopword_encoder = opt.stopword_encoder
        self.stopword_drop_rate = opt.stopword_drop_rate
        stopwords = [
            'y', 'been', 'all', 'shan', 'into', 'him', 'our', 'before', 'so',
            "didn't", 'shouldn', "mustn't", "doesn't", 'yours', 'my', 'and',
            'once', 'did', "hasn't", 'haven', 'yourself', "she's", 'between',
            'up', 'through', 'below', 'is', "haven't", 'theirs', 'me', 'how',
            'now', 'nor', 'were', 'itself', 'just', "isn't", "needn't", 'a',
            'hadn', 'too', 'then', 'from', 'or', 'll', 'ourselves', 'your',
            'she', 'than', 'down', 'has', 'each', 'such', 'during', 'ain',
            'mustn', 'off', 'at', 'will', 'other', 'ma', 'what', 'against',
            'here', "don't", 're', "mightn't", 'who', 'why', "wasn't", 'being',
            'for', 'no', 'an', 'as', 'mightn', "wouldn't", 'above', 'but',
            'out', 'further', 'with', 'having', 'does', 'in', 'of', 'am',
            'you', 's', 'hasn', 'do', 'while', 'few', 'some', 'doesn', 'was',
            'until', 'his', 'their', 'don', 'have', "couldn't", 'wasn', 'that',
            'when', 'they', 'not', "won't", 'ours', 'doing', 'over', 'there',
            'them', 'if', "should've", 'on', 'this', 'its', 'themselves',
            'are', 'which', 'own', 'very', 'd', 'same', 'both', 't', "you'll",
            'm', 'couldn', 'aren', 'hers', 'more', 'i', 'herself', "that'll",
            'yourselves', "you've", 'myself', "aren't", 'o', 'wouldn',
            "weren't", 'be', 'didn', "you're", 'weren', 'about', 'any',
            'because', 'had', 'only', 've', "hadn't", 'himself', 'we',
            "shouldn't", 'can', 'won', "it's", 'isn', 'these', 'should', 'he',
            'those', 'again', 'after', 'by', "shan't", 'where', 'her', 'most',
            'to', 'the', 'it', "you'd", 'under', 'needn', 'whom'
        ]
        self.stopwords = [self.word2id[w]
                          for w in stopwords
                          if w in self.word2id]

        # For context-dependent encoder
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

    def init_state(self, batch_size):
        return Variable(
            torch.zeros(1, batch_size, self.embedding_dim)).to(self.device)

    def forward(self, src_seqs, uniform=False):
        batch_size, max_src_len = src_seqs.size()

        # Keep all by default
        key_prob_over_tokens = torch.ones(src_seqs.size()).to(self.device)
        key_mask = src_seqs.gt(0).long().to(self.device)

        # Uniform encoder
        if self.uniform_encoder or uniform:
            keep_rate_batch = (torch.rand(batch_size)
                               if self.uniform_keep_rate == 0
                               else torch.zeros(batch_size)
                               .fill_(self.uniform_keep_rate))
            rand_prob_over_tokens = keep_rate_batch.unsqueeze(1)\
                .expand_as(src_seqs).to(self.device)
            key_prob_over_tokens = rand_prob_over_tokens

            # Sample from Bernoulli distribution given the probability
            key_mask = torch.distributions.bernoulli.Bernoulli(
                key_prob_over_tokens).sample()

            # Need to mask here for counting keywords
            key_mask.masked_fill_(src_seqs.gt(0).bitwise_not(), -1)
            return key_prob_over_tokens, key_mask

        # Stopword encoder
        elif self.stopword_encoder:
            for i in range(batch_size):
                for j in range(max_src_len):
                    if src_seqs[i][j].item() in self.stopwords:
                        stopword_drop_rate = self.stopword_drop_rate
                        key_prob_over_tokens[i][j] = (1. - stopword_drop_rate)
                        if stopword_drop_rate == 0:
                            stopword_drop_rate = torch.rand(1)
                        if torch.rand(1) < stopword_drop_rate:
                            key_mask[i][j] = 0

            # Need to mask here for counting keywords
            key_mask.masked_fill_(src_seqs.gt(0).bitwise_not(), -1)
            return key_prob_over_tokens, key_mask

        # Context-dependent encoder
        embedded = self.embedding(src_seqs)
        embedded = F.relu(F.dropout(embedded, p=0.1))
        encoder_outputs, _ = self.encoder(embedded)

        # Calculate a probability to keep each token
        key_logit_over_tokens = self.linear(encoder_outputs)\
                                    .view(batch_size, -1)

        masked_key_logit_over_tokens = key_logit_over_tokens.masked_fill(
            src_seqs.gt(0).bitwise_not(), -float('inf'))
        key_prob_over_tokens = torch.sigmoid(masked_key_logit_over_tokens)

        # Sample from Bernoulli distribution given the probability
        key_mask = torch.distributions.bernoulli.Bernoulli(
            key_prob_over_tokens).sample()

        # Mask padding with -1 to mask cross entropy loss later on
        key_mask.masked_fill_(src_seqs.gt(0).bitwise_not(), -1)
        return key_prob_over_tokens, key_mask


class Decoder(nn.Module):
    def __init__(self, opt, tokenizer, word2id, id2word, device):
        super(Decoder, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.word2id = word2id
        self.id2word = id2word
        self.num_tokens = len(word2id)
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.beam_size = opt.beam_size
        self.whitespace = opt.whitespace
        self.capitalization = opt.capitalization
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
        self.attn = nn.Linear(in_features=self.embedding_dim * 2,
                              out_features=self.embedding_dim * 2)
        self.W = nn.Linear(in_features=self.embedding_dim * 4,
                           out_features=self.embedding_dim * 4)
        self.linear = nn.Linear(in_features=self.embedding_dim * 4,
                                out_features=self.num_tokens,
                                bias=True)
        self.linear_copy = nn.Linear(in_features=self.embedding_dim * 6,
                                     out_features=1,
                                     bias=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # NOTE Do not initialize embedding!
        # self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        # self.decoder_embedding.weight.data.uniform_(-initrange, initrange)

        # Encoder and decoder
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # init attn layer
        self.attn.weight.data.uniform_(-initrange, initrange)
        self.attn.bias.data.fill_(0)

        self.W.weight.data.uniform_(-initrange, initrange)
        self.W.bias.data.fill_(0)

        # linear layer
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

        self.linear_copy.weight.data.uniform_(-initrange, initrange)
        self.linear_copy.bias.data.fill_(0)

    def init_state(self, batch_size):
        return Variable(
            torch.zeros(1, batch_size, self.embedding_dim * 2)).to(self.device)

    # Training
    def forward(self, key_seqs, key_lines, trg_seqs):
        batch_size = key_seqs.size(0)

        dyn_word2id, dyn_id2word = self.build_dynamic_vocab(key_lines)
        reindexed_key_seqs = self.reindex(key_seqs,
                                          key_lines,
                                          dyn_word2id).to(self.device)
        encoder_outputs, encoder_last_hidden = self.encode(key_seqs)
        encoder_last_hidden = torch.cat(
            (encoder_last_hidden[0], encoder_last_hidden[1]), dim=1)
        decoder_hidden = (
            encoder_last_hidden.view(1, batch_size, self.embedding_dim * 2),
            self.init_state(batch_size))

        # decoded_batch [batch_size, max_trg_len]
        (decoded_batch, prob_out, attn_out,
         gen_prob, _) = self.teacher_forcing_decode(reindexed_key_seqs,
                                                    trg_seqs,
                                                    dyn_word2id,
                                                    dyn_id2word,
                                                    decoder_hidden,
                                                    encoder_outputs,
                                                    encoder_last_hidden)
        return (prob_out, attn_out, dyn_word2id, dyn_id2word, gen_prob)

    # Training
    def teacher_forcing_decode(self,
                               reindexed_key_seqs,
                               trg_seqs,
                               dyn_word2id,
                               dyn_id2word,
                               decoder_hidden,
                               encoder_outputs,
                               encoder_last_hidden):
        max_key_len = reindexed_key_seqs.size(1)
        batch_size, max_seq_len = trg_seqs.size()
        oov_set = set(dyn_id2word) - set(self.id2word)

        decoder_input = torch.tensor(
            [self.word2id["<sos>"]] * batch_size).to(self.device)
        decoded_batch = torch.zeros((batch_size, max_seq_len))
        prob_out, deco_out, attn_out, gen_prob = [], [], [], []
        for decoder_step in range(max_seq_len):
            (decoder_output, decoder_hidden,
             attn_weights, p_gen) = self.decode(
                decoder_input, decoder_hidden,
                encoder_outputs, encoder_last_hidden,
                dyn_word2id, reindexed_key_seqs)

            prob_out.append(decoder_output.view(batch_size, 1, -1))
            attn_out.append(attn_weights.view(batch_size, 1, max_key_len))
            gen_prob.append(p_gen.view(batch_size, 1, 1))

            decoder_input = trg_seqs[:, decoder_step]  # Teacher forcing
            decoded_batch[:, decoder_step] = decoder_input  # Without oov

        prob_out = torch.cat(prob_out, 1)
        attn_out = torch.cat(attn_out, 1)
        gen_prob = torch.cat(gen_prob, 1)

        return decoded_batch, prob_out, attn_out, gen_prob, None

    # Training or testing
    def greedy_decode(self,
                      reindexed_key_seqs,
                      dyn_word2id,
                      dyn_id2word,
                      decoder_hidden,
                      encoder_outputs,
                      encoder_last_hidden):
        batch_size, max_key_len = reindexed_key_seqs.size()
        max_seq_len = self.max_seq_len
        oov_set = set(dyn_id2word) - set(self.id2word)

        decoder_input = torch.tensor(
            [self.word2id["<sos>"]] * batch_size).to(self.device)
        decoded_batch = torch.zeros((batch_size, max_seq_len))
        prob_out, deco_out, attn_out, gen_prob = [], [], [], []
        for decoder_step in range(max_seq_len):
            (decoder_output, decoder_hidden,
             attn_weights, p_gen) = self.decode(
                decoder_input, decoder_hidden,
                encoder_outputs, encoder_last_hidden,
                dyn_word2id, reindexed_key_seqs)

            prob_out.append(decoder_output.view(batch_size, 1, -1))
            attn_out.append(attn_weights.view(batch_size, 1, max_key_len))
            gen_prob.append(p_gen.view(batch_size, 1, 1))

            # topi [batch_size, 1] (k=1)
            # decoder_input [batch_size]
            # decoded_batch [batch_size, max_seq_len]
            _, topi = decoder_output.data.topk(1)
            decoder_input = topi.view(-1)  # No teacher forcing
            decoded_batch[:, decoder_step] = decoder_input  # With oov

            # Need to replace out-of-vocabulary tokens with <unk>
            for oov in oov_set:
                decoder_input[decoder_input == oov] = self.word2id["<unk>"]

        prob_out = torch.cat(prob_out, 1)
        attn_out = torch.cat(attn_out, 1)
        gen_prob = torch.cat(gen_prob, 1)
        return decoded_batch, prob_out, attn_out, gen_prob, None

    # Testing
    def beam_decode(self,
                    reindexed_key_seqs,
                    dyn_word2id,
                    dyn_id2word,
                    decoder_hidden_,
                    encoder_outputs_,
                    encoder_last_hidden_):
        batch_size, max_key_len = reindexed_key_seqs.size()
        max_seq_len = self.max_seq_len
        oov_set = set(dyn_id2word) - set(self.id2word)

        # Return top 1 prediction for each sequence
        decoded_batch = torch.zeros(batch_size, max_seq_len)
        # Tensors must be in the same size to be concatenated
        prob_out = torch.zeros(batch_size, max_seq_len, len(dyn_word2id))
        attn_out = torch.zeros(batch_size, max_seq_len, max_key_len)
        gen_out = torch.zeros(batch_size, max_seq_len, 1)

        # Return top n predictions for each sequence
        predictions = []  # len(predictions) = batch_size * beam_size
        for i in range(batch_size):
            """
            reindexed_key_seqs [batch_size, max_key_len]
            -> key_seq [1, max_key_len]

            decoder_hidden ([1, batch_size, hidden_size],
                            [1, batch_size, hidden_size])
            -> hidden ([1, 1, hidden_size],
                       [1, 1, hidden_size])

            encoder_outputs [batch_size, max_key_len, hidden_size]
            -> output [1, max_key_len, hidden_size]

            encoder_last_hidden [batch_size, hidden_size]
            -> last_hidden [1, hidden_size]
            """
            # For initialization
            key_seq = reindexed_key_seqs[i].view(1, -1)
            decoder_input = torch.tensor([self.word2id["<sos>"]]).to(self.device)  # noqa
            decoder_hidden = (decoder_hidden_[0][:, i].view(1, 1, -1),
                              decoder_hidden_[1][:, i].view(1, 1, -1))
            encoder_output = encoder_outputs_[i].unsqueeze(0)
            encoder_last_hidden = encoder_last_hidden_[i].view(1, -1)

            # Maintain priority queue of size beam_size
            topk = []
            topk.append(BeamSearchNode(key_seq=key_seq,
                                       decoder_input=decoder_input,
                                       decoder_hidden=decoder_hidden,
                                       encoder_output=encoder_output,
                                       encoder_last_hidden=encoder_last_hidden,
                                       max_seq_len=max_seq_len))

            # Beam search
            for decoder_step in range(max_seq_len):
                new_topk = []
                for candidate in topk:
                    # Skip if already <eos>
                    if candidate.out_seq is not None:
                        if candidate.decoder_input == self.word2id["<eos>"]:
                            new_out_seq = (torch.cat((candidate.out_seq,
                                                      candidate.decoder_input),
                                                     dim=0)
                                           if candidate.out_seq is not None
                                           else candidate.decoder_input.view(1))
                            candidate.out_seq = new_out_seq
                            new_topk.append(candidate)
                            continue
                        elif (candidate.out_seq[-1].item() == self.word2id["<eos>"]
                              or candidate.out_seq.size(0) == max_seq_len):
                            new_topk.append(candidate)
                            continue

                    # If not, predict next token
                    # Replace out-of-vocabulary tokens with <unk>
                    if candidate.decoder_input.item() in self.id2word:
                        decoder_input = candidate.decoder_input
                    else:
                        decoder_input = torch.tensor(
                            [self.word2id["<unk>"]]).to(self.device)

                    (new_decoder_output, new_decoder_hidden, new_attn_weights,
                     new_p_gen) = self.decode(decoder_input,
                                              candidate.decoder_hidden,
                                              candidate.encoder_output,
                                              candidate.encoder_last_hidden,
                                              dyn_word2id,
                                              candidate.key_seq)
                    new_prob_out = (candidate.prob_out +
                                    [new_decoder_output.view(1, 1, -1)])
                    new_attn_out = (candidate.attn_out +
                                    [new_attn_weights.view(1, 1, max_key_len)])
                    new_gen_out = candidate.gen_out + [new_p_gen.view(1, 1, 1)]
                    # Add beam_size candidates per one previous candidate
                    prob, idx = new_decoder_output[0].topk(self.beam_size)
                    for j in range(self.beam_size):
                        # Create new BeamSearchNodes with new info
                        # Do not add <sos> token
                        if candidate.decoder_input.view(1) != self.word2id["<sos>"]:  # noqa
                            new_out_seq = (torch.cat(
                                (candidate.out_seq, candidate.decoder_input),
                                dim=0)
                                if candidate.out_seq is not None
                                else candidate.decoder_input.view(1))
                        else:
                            new_out_seq = None
                        new_score = candidate.score + torch.log(prob[j])
                        new_candidate = BeamSearchNode(
                            key_seq=candidate.key_seq,
                            encoder_output=candidate.encoder_output,
                            encoder_last_hidden=candidate.encoder_last_hidden,
                            decoder_input=idx[j].view(1),
                            decoder_hidden=new_decoder_hidden,
                            out_seq=new_out_seq,
                            score=new_score,
                            prob_out=new_prob_out,
                            attn_out=new_attn_out,
                            gen_out=new_gen_out,
                            max_seq_len=max_seq_len)
                        new_topk.append(new_candidate)

                # Prune out and keep only top beam_size candidates
                new_topk.sort(key=lambda t: t.score, reverse=True)
                topk = new_topk[:self.beam_size]

            # Update with top 1 prediction
            top1 = topk[0]
            top1_prob_out = torch.cat(top1.prob_out, 1).squeeze(0)
            top1_attn_out = torch.cat(top1.attn_out, 1).squeeze(0)
            top1_gen_out = torch.cat(top1.gen_out, 1).squeeze(0)

            decoded_batch[i, :top1.out_seq.size(0)] = top1.out_seq
            prob_out[i, :top1_prob_out.size(0)] = top1_prob_out
            attn_out[i, :top1_attn_out.size(0)] = top1_attn_out
            gen_out[i, :top1_gen_out.size(0)] = top1_gen_out  # Concat timestep

            # Decode top n predictions
            prediction = []
            for topn in topk:
                encoded_line = self.tokenizer.tensor_to_encoded_lines(
                    topn.out_seq, dyn_id2word)[0]
                score = topn.score.item()
                prediction.append((encoded_line, score))
            predictions.append(prediction)

        # Note that decoded_batch contains indices for oov tokens
        return (decoded_batch.to(self.device),
                prob_out.to(self.device),
                attn_out.to(self.device),
                gen_out.to(self.device), predictions)

    # Testing
    def generate(self, key_seqs, key_lines):
        batch_size = key_seqs.size(0)

        dyn_word2id, dyn_id2word = self.build_dynamic_vocab(key_lines)
        reindexed_key_seqs = self.reindex(key_seqs,
                                          key_lines,
                                          dyn_word2id).to(self.device)
        encoder_outputs, encoder_last_hidden = self.encode(key_seqs)
        encoder_last_hidden = torch.cat(
            (encoder_last_hidden[0], encoder_last_hidden[1]), dim=1)
        decoder_hidden = (
            encoder_last_hidden.view(1, batch_size, self.embedding_dim * 2),
            self.init_state(batch_size))

        if self.beam_size > 1:  # Beam search
            # decoded_batch [batch_size, max_seq_len]
            (decoded_batch, prob_out, attn_out,
             gen_prob, predictions) = self.beam_decode(reindexed_key_seqs,
                                                       dyn_word2id,
                                                       dyn_id2word,
                                                       decoder_hidden,
                                                       encoder_outputs,
                                                       encoder_last_hidden)
        else:
            # decoded_batch [batch_size, max_seq_len]
            (decoded_batch, prob_out, attn_out,
             gen_prob, predictions) = self.greedy_decode(reindexed_key_seqs,
                                                         dyn_word2id,
                                                         dyn_id2word,
                                                         decoder_hidden,
                                                         encoder_outputs,
                                                         encoder_last_hidden)

        # Note that decoded_batch contains indices for oov tokens
        return (prob_out, attn_out, dyn_word2id, dyn_id2word,
                gen_prob, predictions)

    def encode(self, key_seqs):
        embedded = self.encoder_embedding(key_seqs)
        embedded = F.relu(F.dropout(embedded, p=0.1))
        encoder_outputs, (encoder_last_hidden, _) = self.encoder(embedded)
        return encoder_outputs, encoder_last_hidden

    def decode(self, decoder_input, decoder_hidden,
               encoder_outputs, encoder_last_hidden,
               dyn_word2id, reindexed_key_seqs):
        batch_size, max_key_len = reindexed_key_seqs.size()

        # embedded [batch_size, hidden_size * 2]
        embedded_ = self.decoder_embedding(decoder_input).view(batch_size, -1)
        embedded_ = F.relu(F.dropout(embedded_, p=0.1))

        # Concat context vector to decoder input
        # encoder_last_hidden [batch_size, hidden_size * 2]
        # embedded [batch_size, hidden_size * 2]
        embedded = torch.cat((embedded_, encoder_last_hidden), dim=1)

        decoder_output, decoder_hidden = self.decoder(
            embedded.view(batch_size, 1, self.embedding_dim * 4),
            decoder_hidden)

        """
        Global attention

        Mask encoder_outputs based on key_lens
        """
        mask = reindexed_key_seqs.gt(0).unsqueeze(
            2).expand_as(encoder_outputs).float()
        encoder_outputs = encoder_outputs.mul(mask)
        # Luong's global attention
        attn_prod = torch.bmm(self.attn(decoder_hidden[0].transpose(0, 1)),
                              encoder_outputs.transpose(1, 2))

        """
        Mask softmax based on src_lens

        attn_weights = F.softmax(attn_prod, dim=2)  # WRONG not masking index 0
        """
        mask_attn = reindexed_key_seqs.gt(0).view(
            batch_size, 1, max_key_len).to(self.device)
        attn_prod.masked_fill_(mask_attn.bitwise_not(), -float('inf'))
        attn_weights = F.softmax(attn_prod, dim=2)
        # attn_weights have nan when there is no input keyword
        # If so, set the weight to be 0 so that attention has no effect
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)
        context = torch.bmm(attn_weights, encoder_outputs)

        hc = torch.cat(
            [decoder_hidden[0].squeeze(0), context.squeeze(1)], dim=1)
        out_hc = self.W(hc)
        decoder_output = self.linear(out_hc)

        # Probability of generating each token
        # decoder_output [64, max_src_len]
        # p_word [64, dyn_vocab_size]
        p_word = torch.zeros([batch_size, len(dyn_word2id)]).to(self.device)
        p_word[:, :len(self.word2id)] = F.softmax(decoder_output, dim=1)

        """
        Copy generator

        p(w) = (1 - p_gen) * p_{copy}(w) + p_gen * p_{word}(w)
        """
        # Probability of generating a token for batch for one timestep
        # decoder_hidden[0] [1, 64, hidden_size * 2]
        # context [64, 1, hidden_size * 2]
        # embedded [batch_size, hidden_size * 4]
        hcx = torch.cat([decoder_hidden[0].squeeze(0),
                         context.squeeze(1), embedded_], dim=1)
        # p_gen [batch_size, 1]
        p_gen = self.sigmoid(self.linear_copy(hcx).view(batch_size, 1))

        # Probability of copying each token
        # attn_weights [64, max_src_len]
        # mul_attn [64, max_src_len]
        # key_map [64, max_src_len, dyn_vocab_size]
        attn_weights = attn_weights.view(batch_size, max_key_len)

        # p_copy [64, dyn_vocab_size]
        p_copy = torch.zeros(batch_size, len(dyn_word2id),
                             device=self.device).scatter_add(
                                 1, reindexed_key_seqs, attn_weights)

        # Probability of predicting each token in dynamic vocab
        # decoder_output [64, dyn_vocab_size]
        decoder_output = (1 - p_gen) * p_copy + p_gen * p_word

        return decoder_output, decoder_hidden, attn_weights, p_gen

    def build_dynamic_vocab(self, encoded_lines):
        tokens = set()
        for encoded_line in encoded_lines:  # Encoded
            tokens.update(self.tokenizer.encoded_line_to_tokens(encoded_line))
        dyn_vocab = [token for token in tokens
                     if token not in self.word2id]

        # Trick to merge two dictionaries without copying
        num_tokens = len(self.word2id)
        dyn_word2id = {**self.word2id,  # noqa
                       **{v: i for i, v in enumerate(dyn_vocab, num_tokens)}}
        dyn_id2word = {**self.id2word,  # noqa
                       **{i: v for i, v in enumerate(dyn_vocab, num_tokens)}}
        return dyn_word2id, dyn_id2word

    def reindex(self, seqs, encoded_lines, dyn_word2id):  # Encoded
        if len(self.word2id) == len(dyn_word2id):
            return seqs

        reindexed_seqs = torch.zeros(seqs.size()).long()
        for i in range(seqs.size(0)):
            tokens = self.tokenizer.encoded_line_to_tokens(encoded_lines[i])
            if len(tokens) > seqs[i].size(0):
                import pdb
                pdb.set_trace()
                print("Error in reindexing; use original seqs")
                print("Line:", encoded_lines[i])
                print("Tokens:", tokens)
                reindexed_seqs[i] = seqs[i]
            else:
                indices = [dyn_word2id.get(token, dyn_word2id["<unk>"])
                           for token in tokens]
                reindexed_seqs[i][:len(indices)] = torch.tensor(indices)
        return reindexed_seqs

    def postprocess(self, key_seqs, key_mask, encoded_lines):
        """
        Receives (key_mask, encoded_lines) or (key_seqs, encoded_lines) pairs.

        (key_mask, encoded_lines): Given key_mask and src_lines, generate
            key_seqs (contains <unk>) and key_lines (contains original tokens).

        (key_seqs, encoded_lines): Given key_seqs and key_lines, simply apply
            postprocessing while keeping all tokens (key_mask=key_seqs.gt(0)).
        """
        if key_mask is None:
            key_mask = key_seqs.gt(0)  # Keep all

        processed_key_seqs = torch.zeros(key_mask.size()).long()
        processed_key_lines = []
        for i, line in enumerate(encoded_lines):  # batch_size
            tokens = self.tokenizer.encoded_line_to_tokens(line)
            processed = []
            for j, token in enumerate(tokens):  # max_seq_len
                if token == "<eos>":  # With <eos>
                    processed.append(token)
                    break
                if self.whitespace == "replace":
                    if key_mask[i][j] == 1:  # Keep
                        if token == "#":
                            if processed and processed[-1] != "#":
                                processed.append(token)
                        else:
                            processed.append(token)
                    else:  # Drop
                        if processed and processed[-1] != "#":
                            # Replace with whitespace
                            processed.append("#")
                elif self.whitespace == "remove":
                    if key_mask[i][j] == 1:  # Keep
                        if token == "#":  # Remove whitespace
                            continue
                        else:
                            processed.append(token)
                    else:  # Drop
                        continue
                elif self.whitespace == "keep":
                    if key_mask[i][j] == 1:  # Keep
                        processed.append(token)
                    else:  # Drop
                        if token == "#":  # Keep whitespace
                            processed.append(token)
                        else:
                            continue
                else:
                    if key_mask[i][j] == 1:  # Keep
                        processed.append(token)
                    else:  # Drop
                        continue
            if processed and processed[-1] == "#":
                processed = processed[:-1]  # Remove trailing whitespace
            if self.capitalization == "remove":
                processed = [token for token in processed
                             if token not in ["<capf>", "<capa>"]]

            seq = self.tokenizer.tokens_to_tensor(processed, self.word2id)
            processed_key_seqs[i][:seq.size(0)] = seq
            encoded_line = self.tokenizer.tokens_to_encoded_line(processed)
            processed_key_lines.append(encoded_line)
        processed_key_seqs = processed_key_seqs.to(self.device)
        return processed_key_seqs, processed_key_lines  # Encoded


class BeamSearchNode(object):
    def __init__(self,
                 key_seq,
                 encoder_output,
                 encoder_last_hidden,
                 decoder_input,
                 decoder_hidden,
                 max_seq_len,
                 out_seq=None,
                 score=0,
                 prob_out=[],
                 attn_out=[],
                 gen_out=[]):
        # Fixed throughout beam search (information from user or encoder)
        self.key_seq = key_seq
        self.encoder_output = encoder_output
        self.encoder_last_hidden = encoder_last_hidden
        self.max_seq_len = max_seq_len

        # Updated in every loop (unless sentence has been terminated by <eos>)
        self.decoder_input = decoder_input
        self.decoder_hidden = decoder_hidden
        self.out_seq = out_seq
        self.score = score

        # Tensors must be in the same size to be concatenated
        self.prob_out = prob_out  # Save decoder_output
        self.attn_out = attn_out  # Save atten_weights
        self.gen_out = gen_out  # Save p_gen


def build_model(opt, tokenizer, word2id, id2word, device):
    decoder = Decoder(opt, tokenizer, word2id, id2word, device).to(device)
    encoder = Encoder(opt, word2id, id2word, device, tokenizer).to(device)
    if opt.lagrangian and (not opt.fix_lambda):
        lambdas = Variable(torch.tensor(opt.init_lambda), requires_grad=True)
    else:
        lambdas = Variable(torch.tensor(opt.init_lambda), requires_grad=False)
    print("\n[Model] Build model.")
    return decoder, encoder, lambdas


def build_models(opt, tokenizer, word2id, id2word, device, model_info):
    encoders, decoders = dict(), dict()
    for name, model_name in model_info.items():
        # Load model
        path_model = os.path.join(opt.root, opt.exp_dir, "model")
        path_load = os.path.join(path_model, "{}".format(model_name))
        try:
            model = (torch.load(path_load) if torch.cuda.is_available()
                     else torch.load(path_load, map_location="cpu"))
        except Exception as e:
            print(f"[Model] FAILED TO LOAD {model_name}.")
            continue

        # Set options for this encoder, decoder pair
        opt_ = copy.deepcopy(opt)
        if "uniformE" in model_name:
            opt_.load_trained_encoder = False
            opt_.uniform_encoder = True
            opt_.uniform_keep_rate = float(re.findall("Kr(.*?)_", model_name)[0])
        elif "stopwordE" in model_name:
            opt_.load_trained_encoder = False
            opt_.stopword_encoder = True
            opt_.stopword_drop_rate = float(re.findall("Dr(.*?)_", model_name)[0])

        encoder = Encoder(opt_, word2id, id2word, device).to(device)
        decoder = Decoder(opt_, tokenizer, word2id, id2word, device).to(device)

        if "uniformE" not in model_name and "stopwordE" not in model_name:
            encoder.load_state_dict(model.get("encoder"))
        decoder.load_state_dict(model.get("decoder"))

        decoders[name] = decoder
        encoders[name] = encoder

    return encoders, decoders


def build_baseline_encoders(opt, tokenizer, word2id, id2word, device):
    encoders = dict()

    stopword_opt = copy.deepcopy(opt)
    stopword_opt.load_trained_encoder = False
    stopword_opt.stopword_encoder = True
    stopword_opt.capitalization = "remove"

    stopword_opt.stopword_drop_rate = 0.5
    encoders["Stop-Dr0.5"] = Encoder(stopword_opt, word2id, id2word, device)

    stopword_opt.stopword_drop_rate = 1
    encoders["Stop-Dr1.0"] = Encoder(stopword_opt, word2id, id2word, device)

    uni_opt = copy.deepcopy(opt)
    uni_opt.load_trained_encoder = False
    uni_opt.uniform_encoder = True

    uni_opt.uniform_keep_rate = 0.1
    encoders["Uni-Kr0.1"] = Encoder(uni_opt, word2id, id2word, device)

    uni_opt.uniform_keep_rate = 0.3
    encoders["Uni-Kr0.3"] = Encoder(uni_opt, word2id, id2word, device)

    uni_opt.uniform_keep_rate = 0.5
    encoders["Uni-Kr0.5"] = Encoder(uni_opt, word2id, id2word, device)

    uni_opt.uniform_keep_rate = 0.7
    encoders["Uni-Kr0.7"] = Encoder(uni_opt, word2id, id2word, device)

    uni_opt.uniform_keep_rate = 0.9
    encoders["Uni-Kr0.9"] = Encoder(uni_opt, word2id, id2word, device)

    for name, encoder in encoders.items():
        print(name)
        encoder = encoder.to(device)

    return encoders

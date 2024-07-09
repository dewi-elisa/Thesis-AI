import configargparse

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import data
import utils
import opts
import dynamic_programming as dp


class Encoder(nn.Module):
    def __init__(self, opt, word2id, id2word, device):
        super(Encoder, self).__init__()
        self.device = device
        self.word2id = word2id
        self.id2word = id2word
        self.opt = opt
        self.num_tokens = len(word2id)
        self.embedding_dim = opt.embedding_dim

        self.embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=0)
        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.embedding_dim,
                               batch_first=True)
        self.w = nn.Parameter(torch.zeros(self.embedding_dim, self.embedding_dim))
        self.linear = nn.Linear(in_features=self.embedding_dim,
                                out_features=1,
                                bias=True)

    def forward(self, tokens):
        # Embed each token
        embedded = self.embedding(tokens)
        embedded = F.relu(F.dropout(embedded, p=0.1))

        # Score each token
        encoder_outputs, _ = self.encoder(embedded)

        if self.opt.segmentation:
            # Score a
            a = self.score_a(encoder_outputs)

            # Sample segmentations
            segmentation = dp.sampling(a)
            segmentation2 = dp.sampling(a)

            # Convert segmentation to mask
            mask = self.segmentation2mask(segmentation)
            mask2 = self.segmentation2mask(segmentation2)

            # Calculate the probability of the segmentation
            log_prob_mask = dp.score_segmentation(a, segmentation) - dp.logsumexp(a)

        else:
            scores = self.linear(encoder_outputs).squeeze(1)

            # Apply sigmoid
            prob_tokens = torch.sigmoid(scores)

            # Sample a mask with Bernoulli
            mask = torch.distributions.bernoulli.Bernoulli(prob_tokens).sample().to(torch.bool)
            mask2 = torch.distributions.bernoulli.Bernoulli(prob_tokens).sample().to(torch.bool)

            # Calculate the log probability of the mask
            log_prob_mask = torch.sum(mask * torch.log(prob_tokens)
                                      + (mask.bitwise_not()) * torch.log(1 - prob_tokens))

        # Apply the mask
        subsentence = tokens[mask]
        subsentence2 = tokens[mask2]

        return subsentence, log_prob_mask, subsentence2

    def score_a(self, H):
        seq_len, hidden_dim = H.shape
        H = torch.cat((torch.zeros(1, hidden_dim), H))
        a = torch.zeros(seq_len + 1, seq_len + 1, 2)
        a[:, :, 1] = H @ self.w @ H.T

        return a

    def segmentation2mask(self, segmentation):
        mask = torch.tensor([], dtype=torch.long)

        for start, end, keep in segmentation:
            n = end - start
            mask = torch.cat((mask, torch.tensor(n * [keep])))

        return mask.to(torch.bool)


class Decoder(nn.Module):
    def __init__(self, opt, word2id, id2word, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word2id = word2id
        self.id2word = id2word
        self.num_tokens = len(word2id)
        self.opt = opt
        self.embedding_dim = opt.embedding_dim

        self.encoder_embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                              embedding_dim=self.embedding_dim,
                                              padding_idx=0, device=self.device)
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
        self.linear_copy = nn.Linear(in_features=self.embedding_dim * 6,
                                     out_features=1,
                                     bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, trg_seqs, decode_function='teacher'):
        # If no keywords were kept, tokens is just <sos> and <eos>
        if torch.equal(tokens, torch.tensor([]).to(self.device)):
            tokens = torch.cat((torch.tensor([self.word2id['<sos>']]),
                                torch.tensor([self.word2id['<eos>']]))).to(self.device)
        # Add <sos> to the tokens
        if torch.ne(tokens[0], torch.tensor(self.word2id['<sos>']).to(self.device)):
            tokens = torch.cat((torch.tensor([self.word2id['<sos>']]).to(self.device), tokens))
        # Add <eos> to the tokens
        if torch.ne(tokens[-1], torch.tensor(self.word2id['<eos>']).to(self.device)):
            tokens = torch.cat((tokens, torch.tensor([self.word2id['<eos>']]).to(self.device)))

        # embed the tokens
        embedded = self.encoder_embedding(tokens)
        embedded = F.relu(F.dropout(embedded, p=0.1))

        # encode the tokens
        encoded, (encoder_hidden, _) = self.encoder(embedded)

        # decode
        batch_size = 1
        last_word = torch.tensor([self.word2id["<sos>"]] * batch_size).to(self.device)
        encoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=-1)
        decoder_hidden = (encoder_hidden.view(1, batch_size, self.embedding_dim * 2),
                          torch.zeros(1, batch_size, self.embedding_dim * 2).to(self.device))

        if decode_function == 'teacher':
            sentence, p_words = self.teacher_forcing_decode(tokens, trg_seqs,
                                                            encoder_hidden, encoded,
                                                            decoder_hidden, last_word)
        elif decode_function == 'greedy':
            sentence, p_words = self.greedy_decode(tokens,
                                                   encoder_hidden, encoded,
                                                   decoder_hidden, last_word)
        else:
            print('Unknown decoding strategy')

        # Calculate the log probability for the sentence
        log_prob_sentence = torch.sum(torch.log(torch.cat(p_words)))

        return sentence, log_prob_sentence

    def teacher_forcing_decode(self, tokens, trg_seqs,
                               encoder_hidden, encoded,
                               decoder_hidden, last_word):
        batch_size = 1
        sentence = [last_word.item()]
        p_words = []

        for word in trg_seqs.squeeze(0):
            p_word, decoder_hidden = self.decode(tokens, encoder_hidden, encoded,
                                                 decoder_hidden, last_word)
            last_word = word.item()
            sentence = sentence + [last_word]
            last_word = torch.tensor([last_word] * batch_size).to(self.device)
            p_words.append(p_word.squeeze(0)[last_word])

        return sentence, p_words

    def greedy_decode(self, tokens, encoder_hidden, encoded, decoder_hidden, last_word):
        batch_size = 1
        sentence = [last_word.item()]
        p_words = []

        for decoder_step in range(self.opt.max_seq_len):
            p_word, decoder_hidden = self.decode(tokens, encoder_hidden, encoded,
                                                 decoder_hidden, last_word)
            last_word = p_word.argmax().item()
            sentence = sentence + [last_word]
            last_word = torch.tensor([last_word] * batch_size).to(self.device)
            p_words.append(p_word.squeeze(0)[last_word])

            # If we generate the end of sentence symbol, stop
            if torch.equal(last_word,
                           torch.tensor([self.word2id["<eos>"]] * batch_size).to(self.device)):
                return sentence, p_words

        return sentence, p_words

    def decode(self, tokens, encoder_hidden, encoded, decoder_hidden, last_word):
        batch_size, max_key_len = 1, tokens.size()

        # Embed the last generated word (or the <sos> symbol if there are none)
        embedded2 = self.decoder_embedding(last_word).view(batch_size, -1)
        embedded2 = F.relu(F.dropout(embedded2, p=0.1))

        # Concatenate the context vector to the embedding
        embedded3 = torch.cat((embedded2, encoder_hidden.view(batch_size, -1)), dim=-1)

        # Decode the embedded token -> generate the next word
        decoded, decoder_hidden = self.decoder(
            embedded3.view(batch_size, 1, self.embedding_dim * 4),
            decoder_hidden)

        '''
        Luong's global attention
        '''
        # First, see which tokens are words
        mask = tokens.gt(0).unsqueeze(1).expand_as(encoded).float()

        # Then, apply the mask to encoded
        encoded = encoded.mul(mask)

        # Put the decoder hidden state in the attention layer and apply to encoded
        attn_prod = torch.bmm(self.attn(decoder_hidden[0].transpose(0, 1)),
                              encoded.transpose(0, 1).unsqueeze(0))

        '''
        mask softmax
        '''
        max_key_len = tokens.size()[0]

        # Look at the tokens that are words
        mask_attn = tokens.gt(0).view(
            batch_size, 1, max_key_len).to(self.device)

        # Replace the tokens that are not a word by -inf
        attn_prod.masked_fill_(mask_attn.bitwise_not(), -float('inf'))

        # Put the attention probabilities in a softmax
        attn_weights = F.softmax(attn_prod, dim=2)

        # If there is a nan in the attention weights, set it to 0
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)

        # Apply the attention weights to encoded
        context = torch.bmm(attn_weights, encoded.unsqueeze(0))

        # Add the attention weights to the last decoder state
        hc = torch.cat(
            [decoder_hidden[0].squeeze(0), context.squeeze(1)], dim=1)
        out_hc = self.W(hc)
        decoder_output = self.linear(out_hc)

        # Initialize p_word
        p_word = torch.zeros([batch_size, len(self.word2id)]).to(self.device)

        # Calculate p_word, the probability of each word to be the next word
        p_word[:, :len(self.word2id)] = F.softmax(decoder_output, dim=1)

        """
        Copy generator
        p(w) = (1 - p_gen) * p_{copy}(w) + p_gen * p_{word}(w)
        """
        # Probability of generating a token
        hcx = torch.cat([decoder_hidden[0].squeeze(0),
                         context.squeeze(1), embedded2], dim=1)
        p_gen = self.sigmoid(self.linear_copy(hcx).view(batch_size, 1))

        # Probability of copying each token
        attn_weights = attn_weights.view(max_key_len)

        p_copy = torch.zeros(len(self.word2id),
                             device=self.device).scatter_add(
                                 0, tokens, attn_weights)

        # Probability of predicting each token in vocab
        p_word = (1 - p_gen) * p_copy + p_gen * p_word

        return p_word, decoder_hidden


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

    encoder = Encoder(opt, word2id, id2word, device).to(device)
    decoder = Decoder(opt, word2id, id2word, device).to(device)

    loaders = data.build_loaders(opt, tokenizer, word2id)
    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    for batch_index, batch in enumerate(train_ae_loader):
        src_seqs, trg_seqs, src_lines, trg_lines = batch

        src_seqs = src_seqs.squeeze(0).to(device)
        trg_seqs = trg_seqs.squeeze(0).to(device)

        # Add <sos> to src_seqs
        src_seqs = torch.cat((torch.tensor([word2id['<sos>']]).to(device), src_seqs))

        keywords, log_q_alpha, _ = encoder(src_seqs)
        predicted, log_p_beta = decoder(keywords, trg_seqs)

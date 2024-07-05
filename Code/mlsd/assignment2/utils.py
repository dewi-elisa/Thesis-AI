# deep learning for structured data
# author: vlad niculae <v.niculae@uva.nl>
# license: MIT

import torch


class EmbedDropout(torch.nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 dropout_idx,
                 word_dropout_p):
        super().__init__()
        self.padding_idx = padding_idx
        self.dropout_idx = dropout_idx
        self.word_dropout_p = word_dropout_p
        self.emb = torch.nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

    def forward(self, seq):

        # apply word dropout: replace p% of tokens with <unk>
        if self.training:
            mask = torch.rand(seq.shape) < self.word_dropout_p
            seq[mask] = self.dropout_idx

        return self.emb(seq)


class ConvItemEncoder(torch.nn.Module):
    """Sequence Convnet that returns a representation of each item (no pooling)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, n_layers=1):
        super().__init__()
        self.layers = [torch.nn.Conv1d(in_channels=input_dim,
                                       out_channels=hidden_dim,
                                       kernel_size=kernel_size,
                                       padding='same',
                                       bias=True)]
        for _ in range(n_layers - 1):
            self.layers.append(
                torch.nn.Conv1d(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=kernel_size,
                                padding='same',
                                bias=True)
            )

    def forward(self, x):
        # convolutions expect the length to be last
        x = x.T

        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)

        x = x.T  # transpose again
        return x

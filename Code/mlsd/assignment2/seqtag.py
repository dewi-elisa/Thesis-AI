# deep learning for structured data
# assignment 2 question 1.2
# original author: vlad niculae <v.niculae@uva.nl>
# license: MIT

# STUDENT ID: 12419273, Dewi Timman

from collections import Counter
import tqdm
import torch

from data_pos import load as load_pos
from data_phon import load as load_phon

from utils import EmbedDropout, ConvItemEncoder
import sequence_dp


def independent_nll(a: torch.Tensor, y: torch.Tensor):
    """Negative log-likelihood loss of a tag sequence, without structure.

    Equivalent to summing up the cross-entropy loss for each position.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    y: torch.Tensor (dtype=long), shape [n_tags]
        Sequence of tag labels, between 0 and n_tags-1.

    Returns
    -------

    loss, scalar
        The total loss of the sequence.
    """

    return torch.nn.functional.cross_entropy(a, y, reduction='sum')


def sequence_nll(a: torch.Tensor, trans: torch.Tensor, y: torch.Tensor):
    """Negative log-likelihood loss of a tag sequence with transition structure

    Uses the Forward algorithm to compute the log-normalizing term over every
    possible tag sequence.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    y: torch.Tensor (dtype=long), shape [n_tags]
        Sequence of tag labels, between 0 and n_tags-1.

    Returns
    -------

    loss, scalar
        The total loss of the sequence.
    """

    score = sequence_dp.score_sequence(a, trans, y)
    logZ = sequence_dp.logsumexp(a, trans)

    return logZ - score


def main():

    # some parts of the preprocessing and even of the encoding model
    # will be different for pos tagging vs speech data.

    TASK = "pos"
    # TASK = "speech"

    load = load_pos if TASK == "pos" else load_phon

    torch.manual_seed(2023)

    hidden_dim = 16
    kernel_size = 3
    # trans_reg = 0.0001
    trans_reg = 0.1
    n_epochs = 10

    # use_transitions = False
    use_transitions = True

    ((x_train, y_train),
     (x_valid, y_valid),
     (x_test, y_test)) = load()

    # Construct vocabulary of input words (POS tagging task only)
    if TASK == "pos":
        # for POS tagging, the input data is a sequence of strings (words).
        # we must encode input data as sequences of ints.
        word_vocab = Counter(w for seq in x_train for w in seq)
        word_vocab = (["<pad>", "<unk>"]
                      + [w for w, count in word_vocab.most_common()
                         if count > 5])
        inv_word_vocab = {w: k for k, w in enumerate(word_vocab)}

        def encode_words(x):
            return torch.tensor(
                [inv_word_vocab.get(w, inv_word_vocab['<unk>']) for w in x],
                dtype=torch.long)

        x_train = [encode_words(x) for x in x_train]
        x_valid = [encode_words(x) for x in x_valid]
        x_test = [encode_words(x) for x in x_test]

        # hyperparameters specific to the POS tagging model
        input_embedding_dim = 16
        word_dropout_p = 0

        embed_layer = EmbedDropout(num_embeddings=len(word_vocab),
                                   embedding_dim=input_embedding_dim,
                                   padding_idx=inv_word_vocab['<pad>'],
                                   dropout_idx=inv_word_vocab['<unk>'],
                                   word_dropout_p=word_dropout_p)

    # for both POS tagging and phoneme prediction, the output sequences must be
    # encoded as sequences of ints.
    tag_vocab = Counter(t for seq in y_train for t in seq)
    tag_vocab = [t for t, count in tag_vocab.most_common()]
    inv_tag_vocab = {t: k for k, t in enumerate(tag_vocab)}

    n_tags = len(tag_vocab)

    def encode_tags(y):
        return torch.tensor([inv_tag_vocab[t] for t in y],
                            dtype=torch.long)

    y_train = [encode_tags(y) for y in y_train]
    y_valid = [encode_tags(y) for y in y_valid]
    y_test = [encode_tags(y) for y in y_test]

    # if too slow, subsample the training data
    # x_train = x_train[::6]
    # y_train = y_train[::6]

    model_pipeline = []

    if TASK == "pos":
        # for POS tagging, the first step is embedding lookup.
        # for speech, there are no input embeddings needed.
        model_pipeline.append(embed_layer)

    # for both POS tagging and speech, we apply a 1d convnet.
    input_dim = input_embedding_dim if TASK == "pos" else x_train[0].shape[-1]
    model_pipeline.append(
        ConvItemEncoder(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size)
    )

    # linear output layer applied at each position
    model_pipeline.append(
        torch.nn.Dropout(p=0.1)
    )
    model_pipeline.append(
        torch.nn.Linear(in_features=hidden_dim,
                        out_features=n_tags,
                        bias=True)
    )

    model = torch.nn.Sequential(*model_pipeline)

    params = list(model.parameters())

    if use_transitions:
        trans = torch.nn.Parameter(torch.randn(n_tags, n_tags))
        params.append(trans)

    def train_pass(xs, ys):
        """Train one pass over the data. (aka one "epoch")"""
        model.train()

        ixs = torch.randperm(len(ys))

        total_loss = 0

        for ix in tqdm.tqdm(ixs):
            x, y = xs[ix], ys[ix]
            out = model(x)

            if use_transitions:
                # structured loss
                loss = sequence_nll(out, trans, y)
                loss += trans_reg * torch.sum(trans ** 2)
            else:
                # unstructured loss: sum up losses for each position
                loss = independent_nll(out, y)

            total_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return total_loss / len(ys)

    def accuracy(xs, ys):
        with torch.no_grad():
            model.eval()

            correct_tags = 0
            correct_seqs = 0
            total_tags = 0
            total_seqs = 0
            for x, y in tqdm.tqdm(zip(xs, ys)):
                out = model(x)

                if use_transitions:
                    _, out_labels = sequence_dp.argmax(out, trans)
                else:
                    out_labels = torch.argmax(out, dim=-1)

                # count number of correct tags and sequences
                for y_tag, out_tag in zip(y, out_labels):
                    if torch.equal(y_tag, out_tag):
                        correct_tags += 1

                if torch.equal(y, out_labels):
                    correct_seqs += 1

                # count total number of tags and sequences
                total_tags += len(y)
                total_seqs += 1

        seq_accuracy = correct_seqs / total_seqs
        tag_accuracy = correct_tags / total_tags

        return seq_accuracy, tag_accuracy

    opt = torch.optim.Adam(params, lr=0.001)

    print("Before training:")
    valid_seq_acc, valid_tag_acc = accuracy(x_valid, y_valid)
    print(f"Valid accuracy: sequence-level {valid_seq_acc:.4f},"
          f"tag-level {valid_tag_acc:.4f}")

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        train_loss = train_pass(x_train, y_train)
        print(f"Training loss {train_loss:.2f}")
        valid_seq_acc, valid_tag_acc = accuracy(x_valid, y_valid)
        print(f"Valid accuracy: sequence-level {valid_seq_acc:.4f},"
              f"tag-level {valid_tag_acc:.4f}")

    if use_transitions:
        torch.save((trans, tag_vocab), "learned_trans.pt")

    # fill in results here in a comment
    # transitions | valid acc (seq)| valid acc (tag micro) |
    # ------------|----------------|-----------------------|
    # no          | 11.10 %        | 77.57 %               |
    # yes         | 10.80 %        | 75.35 %               |

    # The accuracy without transitions is higher than the accuracy with transitions.
    # In addition, the sequence accuracy is lower than the tag accuracy. This is also
    # what I would expect, since it is more difficult to have a sequence of tags
    # predicted correct than to have just one tag predicted correct.

    # Comment about the plot
    # Positive example: Det - Noun
    # Negative example: Noun - Noun


if __name__ == '__main__':
    main()

# deep learning for structured data
# assignment 2 question 1.1
# original author: vlad niculae <v.niculae@uva.nl>
# license: MIT

# STUDENT ID: 12419273, Dewi Timman

import torch
import itertools


def score_sequence(a: torch.Tensor,
                   trans: torch.Tensor,
                   y: torch.Tensor):
    """Compute the score of a sequence y, under given model.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    trans: torch.Tensor, shape [n_tags, n_tags]
        Pairwise transition scores: trans[s, t] is the score
        rewarded for observing a transition from tag s to tag t.

    y: torch.Tensor (dtype=long), shape [seq_len]
        Sequence of tag labels, between 0 and n_tags-1.


    Returns
    -------

    score, scalar
        The score of the tag sequence y.

    Examples
    --------

    If a and trans are full of zeros, then any sequence score is zero.
    """

    seq_len, n_tags = a.shape
    score = a[0, y[0]]

    for j in range(1, seq_len):
        score = score + a[j, y[j]] + trans[y[j-1], y[j]]

    return score


def argmax_brute_force(a: torch.Tensor, trans: torch.Tensor):
    """Find the maximum-scoring sequence under given model.

    Brute force approach by explicit enumeration.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    trans: torch.Tensor, shape [n_tags, n_tags]
        Pairwise transition scores: trans[s, t] is the score
        rewarded for observing a transition from tag s to tag t.

    Returns
    -------

    score: scalar,
        The score of the highest-scoring sequence

    y: torch.Tensor (dtype=long), shape [seq_len]
        The highest-scoring sequence of tag labels.
    """

    seq_len, n_tags = a.shape

    # generate all possible combinations
    # hint: read the documentation of the python itertools module
    seqs = list(itertools.product(range(n_tags), repeat=seq_len))

    best_score = score_sequence(a, trans, torch.Tensor(seqs[0]).long())
    best_seq = seqs[0]

    for seq in seqs:
        score = score_sequence(a, trans, torch.Tensor(seq).long())
        if score > best_score:
            best_score = score
            best_seq = seq

    return best_score, best_seq


def logsumexp_brute_force(a: torch.Tensor, trans: torch.Tensor):
    """Compute sequence normalizing constant (log-sum-exp) under given model.

    Brute force approach by explicit enumeration.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    trans: torch.Tensor, shape [n_tags, n_tags]
        Pairwise transition scores: trans[s, t] is the score
        rewarded for observing a transition from tag s to tag t.

    Returns
    -------

    log_normalizer: scalar,
        The log sum exp of the score of every possible sequence.
    """

    seq_len, n_tags = a.shape

    # generate all possible combinations
    seqs = list(itertools.product(range(n_tags), repeat=seq_len))

    scores = torch.Tensor([score_sequence(a, trans, torch.Tensor(seq).long()) for seq in seqs])

    return torch.logsumexp(scores, dim=0)


def logsumexp(a: torch.Tensor, trans: torch.Tensor):
    """Compute normalizing constant (log-sum-exp) under given model.

    Efficient implementation using the forward algorithm.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    trans: torch.Tensor, shape [n_tags, n_tags]
        Pairwise transition scores: trans[s, t] is the score
        rewarded for observing a transition from tag s to tag t.

    Returns
    -------

    log_normalizer: scalar,
        The log sum exp of the score of every possible sequence.

    """
    seq_len, n_tags = a.shape

    Q = torch.zeros(seq_len, n_tags)
    Q[0, :] = a[0, :]

    for i in range(1, seq_len):
        # forward algorithm
        scores = Q[i - 1, :] + (a[i, :] + trans).T
        Q[i, :] = torch.logsumexp(scores.T, dim=0)

    return torch.logsumexp(Q[seq_len - 1, :], dim=0)


def argmax(a: torch.Tensor, trans: torch.Tensor):
    """Find the maximum-scoring sequence under given model.

    Efficient implementation using the Viterbi algorithm.

    Parameters
    ----------

    a: torch.Tensor, shape [seq_len, n_tags]
        Unary scores for each tag at each location in the sequence.

    trans: torch.Tensor, shape [n_tags, n_tags]
        Pairwise transition scores: trans[s, t] is the score
        rewarded for observing a transition from tag s to tag t.

    Returns
    -------

    score: scalar,
        The score of the highest-scoring sequence

    y: torch.Tensor (dtype=long), shape [seq_len]
        The highest-scoring sequence of tag labels.
    """
    seq_len, n_tags = a.shape

    # viterbi forward
    M = torch.zeros(seq_len, n_tags)
    M[0, :] = a[0, :]
    pi = torch.zeros(seq_len, n_tags)

    for j in range(1, seq_len):
        scores = M[j - 1, :] + (a[j, :] + trans).T
        max = torch.max(scores, keepdim=True, dim=1)
        M[j, :], pi[j, :] = max.values.T[0], max.indices.T[0]

    best_score = torch.argmax(M[seq_len - 1, :])
    score = M[seq_len - 1, best_score]

    # viterbi backward
    seq = torch.zeros(seq_len).long()
    seq[seq_len - 1] = best_score
    j = seq_len - 2
    while j >= 0:
        seq[j] = pi[j+1, seq[j+1]]
        j -= 1

    return score, seq


def main():

    # torch.manual_seed(2022)
    a = 0.01 * torch.randn(5, 3)
    trans = torch.randn(3, 3)

    print(argmax_brute_force(a, trans))
    print(logsumexp_brute_force(a, trans))

    print(argmax(a, trans))
    print(logsumexp(a, trans))


if __name__ == '__main__':

    main()

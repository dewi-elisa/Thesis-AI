import torch
import itertools


def valid_segmentation(a, segmentation):
    """Check if the segmentation is valid.

    Parameters
    ----------
    a: torch.Tensor, shape [seq_len+1, seq_len+1]
        Scores for each segment.

    segmentation: torch.Tensor (dtype=long), shape [seq_len+1]
        Sequence of segments, between 0 and seq_len.

    Returns
    -------
    bool
        True if the segmentation is valid, False otherwise.
    """
    _, seq_len_plus_1, _ = a.shape

    # check if the first segmentation starts at 0
    if segmentation[0][0] != 0:
        return False

    # check is the last segmentation ends at the end
    if segmentation[-1][1] != seq_len_plus_1 - 1:
        return False

    # check it the segments follow each other
    current = 0
    for start, end, _ in segmentation:
        if start == end:
            return False
        if start != current:
            return False
        if end <= current:
            return False
        current = end

    return True


def score_segmentation(a, segmentation):
    """Compute the score of a segmentation, under given model.

    Parameters
    ----------
    a: torch.Tensor, shape [seq_len+1, seq_len+1]
        Scores for each segment.

    segmentation: torch.Tensor (dtype=long), shape [seq_len+1]
        Sequence of segments, between 0 and seq_len.

    Returns
    -------
    score, scalar
        The score of the tag sequence y.
    """
    score = torch.tensor(0)

    for start, end, keep in segmentation:
        score = score + a[start, end, keep]

    return score


def generate_all_segmentations(seq_len_plus_1):
    """Check if the segmentation is valid.

    Parameters
    ----------
    seq_len_plus_1: torch.Tensor

    Returns
    -------
    segmentations: torch.Tensor
        All possible segmentations
    """
    # generate all possible segments
    segments = list(itertools.combinations(range(seq_len_plus_1), 2))
    segments = list(itertools.product(segments, [True, False]))

    # generate all possible combinations
    segmentations = []
    for r in range(1, seq_len_plus_1):
        new_segmentations = itertools.product(segments, repeat=r)
        for x in list(new_segmentations):
            segmentation = torch.tensor([], dtype=torch.long)
            for (start, end), keep in x:
                segment = torch.tensor([[start, end, keep]])
                segmentation = torch.cat((segmentation, segment))
            if valid_segmentation(a, segmentation):
                segmentations.append(segmentation)

    return segmentations


def argmax_brute_force(a: torch.Tensor):
    """Find the maximum-scoring segmentation using brute force.

    Parameters
    ----------
    a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
        Tensor of scores, defined as follows.
         - a[i, j, 0] is the score for leaving out segment i-j.
         - a[i, j, 1] is the score for keeping segment i-j.

    Returns
    -------
    score: scalar,
        Score of the highest-scoring segmentation

    y: torch.Tensor (dtype=long), shape [seq_len]
        The highest-scoring segmentation.
    """
    _, seq_len_plus_1, _ = a.shape

    segmentations = generate_all_segmentations(seq_len_plus_1)

    best_score = score_segmentation(a, segmentations[0])
    best_segmentation = segmentations[0]

    for segmentation in segmentations:
        score = score_segmentation(a, segmentation)

        if score > best_score:
            best_score = score
            best_segmentation = segmentation

    return best_segmentation, best_score


def argmax(a: torch.Tensor):
    """Find the maximum-scoring segmentation using Viterbi.

    Parameters
    ----------
    a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
        Tensor of scores, defined as follows.
         - a[i, j, 0] is the score for leaving out segment i-j.
         - a[i, j, 1] is the score for keeping segment i-j.

    Returns
    -------
    score: scalar,
        Score of the highest-scoring segmentation

    y: torch.Tensor (dtype=long), shape [seq_len]
        The highest-scoring segmentation.
    """
    _, seq_len_plus_1, _ = a.shape

    a_keep = torch.ge(a[:, :, 1], a[:, :, 0]).int()
    a_score = torch.amax(a, 2)

    m = torch.zeros(seq_len_plus_1)
    pi = torch.zeros(seq_len_plus_1, dtype=torch.long)
    pi_keep = torch.zeros(seq_len_plus_1, dtype=bool)

    pi[0] = 0
    pi_keep[0] = True

    for i in range(1, seq_len_plus_1):
        keep = []
        scores = []
        for j in range(i):
            scores.append(m[j] + a_score[j, i])
            keep.append(a_keep[j, i])
        max = torch.max(torch.tensor(scores), keepdim=True, dim=0)
        m[i] = max.values
        pi[i] = max.indices[0]
        pi_keep[i] = keep[max.indices[0]]

    segmentation = torch.tensor([], dtype=torch.long)
    i = seq_len_plus_1 - 1
    while i > 0:
        segment = torch.tensor([[pi[i], i, pi_keep[i]]])
        segmentation = torch.cat((segment, segmentation))
        i = pi[i]

    score = score_segmentation(a, segmentation)

    return segmentation, score


def logsumexp_brute_force(a: torch.Tensor):
    """Compute sequence normalizing constant (log-sum-exp) using brute force.

    Parameters
    ----------
    a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
        Tensor of scores, defined as follows.
         - a[i, j, 0] is the score for leaving out segment i-j.
         - a[i, j, 1] is the score for keeping segment i-j.

    Returns
    -------
    log_normalizer: scalar,
        The log sum exp of the score of every possible segmentation.
    """
    _, seq_len_plus_1, _ = a.shape

    segmentations = generate_all_segmentations(seq_len_plus_1)

    scores = torch.tensor([score_segmentation(a, segmentation) for segmentation in segmentations])

    return torch.logsumexp(scores, dim=0)


def logsumexp(a: torch.Tensor):
    """Compute normalizing constant (log-sum-exp) using Forward.

    Parameters
    ----------
    a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
        Tensor of scores, defined as follows.
         - a[i, j, 0] is the score for leaving out segment i-j.
         - a[i, j, 1] is the score for keeping segment i-j.

    Returns
    -------
    log_normalizer: scalar,
        The log sum exp of the score of every possible segmentation.
    """
    _, seq_len_plus_1, _ = a.shape

    q = torch.zeros(seq_len_plus_1)

    for i in range(1, seq_len_plus_1):
        x = torch.tensor([])
        for j in range(i):
            x = torch.cat((q[j] + a[j, i, :], x))
        q[i] = torch.logsumexp(x, dim=0)

    return q[-1]


if __name__ == '__main__':
    a = torch.tensor([[[0, 0], [10, 5], [6, 9]],
                      [[0, 0], [0, 0], [7, 8]],
                      [[0, 0], [0, 0], [0, 0]]])
    print(a.size())

    print(argmax_brute_force(a))
    print(argmax(a))

    print()
    print(logsumexp_brute_force(a))
    print(logsumexp(a))

# deep learning for structured data
# assignment 2 question 3
# original author: vlad niculae <v.niculae@uva.nl>
# license: MIT

# STUDENT ID: 12419273, Dewi Timman

import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt


def max_matching_lp(G, integer=False):
    """Find maximal matching in a weighted graph G via linear programming.

    Parameters
    ----------

    G: networkx.Graph,
        The input graph. Should be undirected, but not necessarily weighted.

    integer: bool,
        Whether to impose integer constraints on the solution or allow
        fractional solutions.

    Returns
    -------

    matching:, np.array
        A vector of the same size as the number of edges in the input graph,
        where matching[i] == 1 if the ith edge is in the matching, 0 otherwise.
        (If integer=False, fractional values indicate approximate solutions.)

    """

    # n = G.number_of_nodes()
    m = G.number_of_edges()

    # create a cvxpy variable of dimension num_edges:
    matching = cp.Variable(shape=(m,), integer=integer)

    edge_weights = np.array([data['weight']
                             for _, _, data in G.edges(data=True)])

    # compute the linear objective (sum of weights of selected edges.)
    # Hint: use a dot product.
    objective = edge_weights @ matching

    constraints = [
        # add here the box constraints on the variables.
        matching >= 0,
        matching <= 1
    ]

    # make per-node constraints
    # For each node, add a constraint that at most one of the edges incident to
    # it is selected.
    for node in G.nodes():
        indices = []
        for i, edge in enumerate(G.edges()):
            if edge in G.edges(node):
                indices.append(i)
        constraints.append(cp.sum(matching[indices]) <= 1)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(verbose=True)

    return matching.value


def max_matching_bipartite_lp(X, integer=False):
    """Find maximal matching in a weighted bipartite graph G.

    Parameters
    ----------

    X, np.array, shape [n1, n2]
        The dense adjacency matrix of the input graph.

    integer: bool,
        Whether to impose integer constraints on the solution or allow
        fractional solutions.

    Returns
    -------

    M:, np.array
        Array of the same size as X, where M[u, v] == 1 if the edge from u to v
        is in the matching, 0 otherwise. (If integer=False, fractional values
        indicate approximate solutions.)

    """

    transpose = X.shape[1] < X.shape[0]
    if transpose:
        X = X.T

    # Here, since we know the graph is dense,
    # we can more easily deal with edges as an n-by-m matrix.
    M = cp.Variable(X.shape, integer=integer)

    # the objective is the "matrix dot product" between M and X.
    # implement it (one way is to use cp.multiply and cp.sum)
    objective = cp.sum(cp.multiply(M, X))

    constraints = [
        # add all the constraints here
        M >= 0,
        cp.sum(M, axis=1) == 1,
        cp.sum(M, axis=0) <= 1
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(verbose=True)

    res = M.value
    if transpose:
        res = res.T

    return res


def main():

    # graph matching
    G = nx.karate_club_graph()
    matching_weights = max_matching_lp(G)

    matching_weights = matching_weights.round(2)  # for plotting

    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, edge_color=matching_weights,
            edge_cmap=plt.cm.PuOr_r,
            edge_vmin=-1, edge_vmax=1)
    nx.draw(G, pos=pos, alpha=0.2, width=3)
    plt.show()

    # bipartite matching
    rng = np.random.default_rng(seed=2022)
    X = 0.1 * rng.standard_normal((17, 13))
    M = max_matching_bipartite_lp(X)

    _, (ax1, ax2) = plt.subplots(1, 2)
    im_X = ax1.imshow(X)
    plt.colorbar(im_X, ax=ax1)
    im_res = ax2.imshow(M)
    plt.colorbar(im_res, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()

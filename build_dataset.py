import numpy as np


def adjacency_label(x):

    """ Draw a random labeled graph with respect to the input x
    """

    n_nodes = 40

    A = np.zeros((n_nodes, n_nodes))
    P = np.zeros((n_nodes, n_nodes))
    p = 0.9  # probability connexion in cluster
    q = 0.01  # probability connexion between cluster

    if x < 1:
        n_blocks = 2
        eps = 0
    else:
        n_blocks = int(x) + 1  # target number of blocks
        eps = x - int(x)  # interpolation value

    # Linear interpolation
    r = (q - p) * eps + p

    # Number of nodes in each target blocks
    N = [n_nodes]
    b0 = -1
    while len(N) < n_blocks:
        n1 = np.max(N) // 2
        n2 = np.max(N) - n1
        b0 = int(np.argmax(N))
        N[b0] = n1
        N.append(n2)

    # Random number of additional nodes for each blocks
    for i in range(n_blocks):
        N[i] += np.random.randint(0, 5)

    # Block of each nodes
    B = []
    idx = 0
    for i in range(n_blocks):
        B += N[i] * [i]
        idx += N[i]

    B = np.random.permutation(B)  # no canonical order known, i.e. relationship between nodes

    # Blocks that are separating from each other
    i1 = b0
    i2 = len(N) - 1

    # Compute adjacency matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                if B[i] == B[j]:
                    rand = p
                else:
                    if (B[i], B[j]) == (i1, i2) or (B[j], B[i]) == (i1, i2):
                        rand = r
                    else:
                        rand = q
                u = np.random.random()
                if u < rand:
                    A[i, j] = 1
                P[i, j] = rand

    # Compute feature matrix
    F = []
    for i in range(n_nodes):
        if B[i] == i2:
            u = np.random.random()
            if u < eps:
                F.append(i2)
            else:
                F.append(i1)
        else:
            F.append(B[i])

    F = np.array(F).reshape((-1, 1))

    return A, P, F


def create_data_train(n):

    """ Draw a training dataset of size n
    """

    X_tr = np.array(list(5 * np.random.random(n) + 0.5))
    Y_tr = []
    for x in X_tr:
        C_true, _, F_true = adjacency_label(x)
        Y_tr.append([C_true, F_true])
    X_tr = np.array(X_tr).reshape((-1, 1))

    return X_tr, Y_tr


def create_data_test():

    """ Draw the test dataset
    """

    X_te = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5., 5.5]

    Y_te = []
    for x in X_te:
        C_true, _, F_true = adjacency_label(x)
        Y_te.append([C_true, F_true])
    X_te = np.array(X_te).reshape((-1, 1))

    return X_te, Y_te

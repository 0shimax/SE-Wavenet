import numpy as np
import itertools


def add_diff_features(features):
    seq_len, n_dim = features.shape
    comb = list(itertools.combinations(range(n_dim), 2))
    add_mat = np.empty([seq_len, len(comb)], dtype=np.float32)
    for i_add, cmb in enumerate(comb):
        add_mat[:, i_add] = features[:, cmb[0]] - features[:, cmb[1]]
    return np.concatenate((features, add_mat), axis=1)
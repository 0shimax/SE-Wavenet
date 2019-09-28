import torch
import numpy as np


def as_seaquence(seq, ahead=10):
    """
    input:
        seq: (n_time,)
            we assume seq is applied argmax function
    """

    seq_len = seq.shape[0]
    for i in range(seq_len - ahead - 1):
        if seq[i] == seq[i+1]:
            continue

        for i_search in range(i+2, i+ahead+1):
            if seq[i] == seq[i_search]:
                seq[i+1:i_search] = seq[i]
                break
    return seq


if __name__=='__main__':
    seq = np.array([[[1]]*10 + [[0]]*2 + [[1]]*10 + [[3]]*4 + [[1]] + [[3]]*2]).T
    print(seq.ravel())
    # if ahead is 3, search index to i+3
    print(as_seaquence(seq, ahead=3))

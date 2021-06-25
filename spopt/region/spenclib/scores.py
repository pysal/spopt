import numpy as np


def boundary_fraction(W, labels, X=None):
    """"""
    boundary = 0
    for row, own_label in zip(W, labels):
        neighbor_labels = labels[row.nonzero()[-1]]
        boundary += (neighbor_labels != own_label).any().astype(int)
    return boundary / W.shape[0]


def boundary_score(W, labels, X=None):
    """
    Returns a version of boundary_fraction unbounded on the negative end using
    the log of the fraction:

    np.log(boundary_fraction(W, labels))

    This is solely for testing purposes.
    """
    return np.log(boundary_fraction(W, labels, X=None))

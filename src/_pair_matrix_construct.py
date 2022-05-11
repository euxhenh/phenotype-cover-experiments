from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
from sklearn.utils.validation import check_X_y, indexable, column_or_1d
from tqdm import tqdm

from ._operations import group_by


def _pairwise_differences(
        X, y,
        *,
        classes=None,
        ordered=False,
        operation=lambda x: x.mean(axis=0)):
    """
    Given an data matrix X, construct a matrix M of shape
    (n * (n-1) / 2, X.shape[1]) where n is the number of classes in y.
    The (i*j, g) entry of M corresponds to the average expression of feature g
    in group i - average expression of feature g in group j, clipped
    at a minimum of 0.

    Returns M and a dictionary of mappings: label, label -> index.

    Parameters
    _________
    X: np.ndarray of shape (n_samples, n_features)
    y: np.ndarray of shape (n_samples,)
    classes: np.ndarray or None, unique class labels in y
    ordered: bool, if True will construct a matrix of ordered
        pairwise differences. In this case the shape of M is
        (n * (n-1), X.shape[1]).
    operation: callable, operation to use when constructing the class vector.
    """
    if classes is None:
        classes = np.unique(y)

    n_classes = len(classes)
    # All pairwise combinations
    n_class_pairs = n_classes * (n_classes - 1) // 2

    # Cache the average vector of each class
    class_averages = group_by(
        X, y, category_orders=classes, operation=operation)

    # Compute the actual pairwise differences
    M = np.zeros((n_class_pairs * (1 if not ordered else 2), X.shape[1]))
    index_to_pair_dict = {}

    # Make sure to use range(n_classes) when indexing instead of classes,
    # to allow for arbitrary class labels.
    for index, (i, j) in enumerate(combinations(range(n_classes), 2)):
        difference = class_averages[i] - class_averages[j]
        if ordered:
            # Clip negative values to 0
            # Assign i - j to index and j - i to index + n_class_pairs
            M[index] = np.clip(difference, 0, None)
            index_to_pair_dict[index] = (i, j)
            M[index + n_class_pairs] = np.clip(-difference, 0, None)
            index_to_pair_dict[index + n_class_pairs] = (j, i)
        else:
            M[index] = np.abs(difference)
            index_to_pair_dict[index] = (i, j)

    return M, index_to_pair_dict

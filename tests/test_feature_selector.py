import unittest
from itertools import combinations

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import wasserstein_distance as wd
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from src._pair_matrix_construct import (_pairwise_differences,
                                        _pairwise_wasserstein,
                                        _wasserstein_parallel)
from src.feature_selector import GreedyCoverSelector


class TestSelectors(unittest.TestCase):
    def test1(self):
        X = np.array([
            [0, 1, 2, 4, 1, 4],
            [2, 5, 1, 3, 4, 1],
            [3, 4, 1, 2, 1, 1],
            [4, 6, 7, 1, 3, 5],
            [6, 1, 3, 4, 5, 1]
        ], dtype=float)
        labels = np.array([0, 0, 1, 1, 2], dtype=int)
        M, mapping = _pairwise_differences(X, labels)

        assert_allclose(M, np.array([
            [2.5, 2, 2.5, 2, 0.5, 0.5],  # 0 - 1
            [5, 2, 1.5, 0.5, 2.5, 1.5],  # 0 - 2
            [2.5, 4, 1, 2.5, 3, 2]  # 1 - 2
        ]))

    def test2(self):
        X = np.array([
            [0, 1, 2, 4, 1, 4],
            [2, 5, 1, 3, 4, 1],
            [3, 4, 1, 2, 1, 1],
            [4, 6, 7, 1, 3, 5],
            [6, 1, 3, 4, 5, 1]
        ], dtype=float)
        labels = np.array([0, 0, 1, 1, 2], dtype=int)
        M, mapping = _pairwise_differences(X, labels, ordered=True)

        assert_allclose(M, np.array([
            [0, 0, 0, 2, 0.5, 0],  # 0 - 1
            [0, 2, 0, 0, 0, 1.5],  # 0 - 2
            [0, 4, 1, 0, 0, 2],  # 1 - 2
            [2.5, 2, 2.5, 0, 0, 0.5],  # 1 - 0
            [5, 0, 1.5, 0.5, 2.5, 0],  # 2 - 0
            [2.5, 0, 0, 2.5, 3, 0]  # 2 - 1
        ]))

    def test_greedy_cover_selector(self):
        gcs = GreedyCoverSelector()

        X = np.array([
            [1, 0, 0, 1, 2],
            [0, 2, 1, 0, 3],
            [1, 2, 0, 3, 0],
            [0, 2, 1, 3, 0],
            [1, 0, 0, 4, 0]
        ])
        y = np.array([0, 0, 1, 1, 1])

        gcs.fit_select(X, y, coverage=2)
        sfm = SelectFromModel(gcs, threshold=0.5, prefit=True)

        assert_allclose(sfm.get_support(indices=True), np.array([3, 4]))
        assert_allclose(sfm.transform(X), X[:, np.array([3, 4])])

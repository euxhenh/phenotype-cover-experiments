"""
Implements a fast neighbors search class based on approximate neighbors
using the faiss package. If the number of points is small, switches to
exact nearest neighbors computation using sklearn.
"""

import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import scipy.sparse as sp

from .utils.validation import _validate_algorithm, _validate_n_neighbors

THRESHOLD = 1_000


class FastNeighbor:
    def __init__(self, n_neighbors=15, algorithm='auto', n_jobs=-1):
        """
        Args:
            n_neighbors: int, number of nearest neighbors to compute
            algorithm: str, algorithm to use: 'exact', 'approx', or 'auto'
            n_jobs: int, number of threads to use
        """
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.algorithm = _validate_algorithm(algorithm)
        self.n_jobs = n_jobs
        self._is_approx = None
        self._X = None

    @property
    def is_auto(self):
        return self.algorithm == 'auto'

    @property
    def is_approx(self):
        return self.algorithm == 'approx'

    @property
    def n_samples_(self):
        if hasattr(self, '_n_samples'):
            raise ValueError("No data has been fit.")
        return self._n_samples

    @property
    def n_features_(self):
        if hasattr(self, '_n_features'):
            raise ValueError("No data has been fit.")
        return self._n_features

    def _fit_approx(self, X):
        """
        Fit the data X using faiss.
        """
        self._X = np.ascontiguousarray(X.astype(np.float32))

        self.index = faiss.IndexHNSWFlat(X.shape[1], min(15, X.shape[0]))
        self.index.add(self._X)

        self._is_approx = True

    def _fit_exact(self, X):
        """
        Fit the data X using sklearn.
        """
        self.index = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      n_jobs=self.n_jobs)

        self._X = X
        self.index.fit(self._X)

        self._is_approx = False

    def fit(self, X):
        """
        Determine which fit function to call.
        """
        if sp.issparse(X):
            raise ValueError("Cannot work with sparse matrices.")

        self._n_samples = X.shape[0]
        self._n_features = X.shape[1]
        if (self.is_auto and X.shape[0] > THRESHOLD) or self.is_approx:
            self._fit_approx(X)
        else:
            self._fit_exact(X)

    def predict(self, X=None):
        """
        Determines the appropriate neighbors function to use
        and also corrects for `self` if X is None.
        If X is None, will return the neighbors of the fitted X.
        """
        if self._is_approx is None:
            raise ValueError("Please run `fit` first.")

        if sp.issparse(X):
            raise ValueError("Cannot work with sparse matrices.")

        # If self._exclude_self is True, the number of neighbors is
        # effectively increased by one to allow correction.

        if X is None:
            X_to_predict = self._X
        else:
            X_to_predict = X

        if self._is_approx:  # If using approximate neighbors
            X_to_predict = np.ascontiguousarray(
                X_to_predict.astype(np.float32))
            # If X is None, than we are predicting the fitted data.
            # In this case, increase the number of neighbors by 1,
            # to correct for self.
            _, indices = self.index.search(
                X_to_predict, self.n_neighbors + (X is None))

            if X is None:
                indices_corrected = np.empty(
                    (X_to_predict.shape[0], self.n_neighbors), dtype=int)
                for i in tqdm(range(X_to_predict.shape[0])):
                    # First remove `self`
                    filtered_neighs = indices[i][indices[i] != i]
                    # Take first n_neighbors in case `self` was not in the array
                    # which is not really supposed to happen
                    indices_corrected[i] = filtered_neighs[:self.n_neighbors]
                indices = indices_corrected
        else:
            indices = self.index.kneighbors(
                X_to_predict,
                n_neighbors=self.n_neighbors + (X is None),
                return_distance=False)

            # Since we computed the exact neighbors here, than the first
            # column will be 0's so we remove it.
            # In case there are multiple points that equal 0, the first
            # index may not correspond to `self`, but we can still remove it
            # as the distances are the same.
            if X is None:
                indices = indices[:, 1:]

        assert indices.shape == (X_to_predict.shape[0], self.n_neighbors)
        self.indices_ = indices
        return indices

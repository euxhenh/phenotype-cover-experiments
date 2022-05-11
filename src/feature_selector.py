import gc
import os
from abc import abstractmethod
from itertools import combinations

import diffxpy.api as de
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from ._base import FeatureSelector
from ._logger import logger
from ._pair_matrix_construct import _pairwise_differences
from .gci_wrapper import GCIPython, GCIWrapper
from .neighbors import FastNeighbor
from .utils.plots import plot_heatmap


class SetCoverSelectorBase(FeatureSelector):
    """
    Constructs a base class for set cover selector algorithms.
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            operation=lambda x: x.mean(axis=0)):
        self.ordered = ordered
        self.verbose = verbose
        self.operation = operation

    def fit(self, X, y, *, Mpath=None):
        """
        Given a matrix X and a class label vector y, construct a matrix of
        class pairwise differences for every gene and use it to initialize
        a GreedyCoverInstance via a GCIWrapper.

        Parameters
        __________
        X: ndarray, data matrix, shape (n_samples, n_features)
        y: ndarray, class label vector, shape (n_samples,)
        multiplier: int or None, if not None will be used to multiply
            the pairwise differences to allow for finer resolution
            of the element multiplicities in the greedy cover algorithm.
        Mpath: None or str
            Path to a precomputed pairwise matrix.
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            ensure_min_samples=2,
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )

        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ <= 1:
            raise ValueError("At least 2 class labels must be given.")

        self.n_class_pairs_ = self.n_classes_ * (self.n_classes_ - 1)
        if not self.ordered:
            self.n_class_pairs_ //= 2

        # M_ has shape (n_class_pairs, n_features_)
        if Mpath is None:
            self.M_, self.index_to_pair_ = _pairwise_differences(
                X, y, classes=self.classes_, ordered=self.ordered,
                operation=self.operation)
        else:
            self.M_ = np.load(Mpath)
            if len(self.M_) != self.n_class_pairs_:
                raise ValueError(
                    "Mismatch between the number of class pairs "
                    "and the shape of the pairwise matrix M."
                )
            self.index_to_pair_ = {
                index: (i, j)
                for index, (i, j)
                in enumerate(combinations(range(self.n_classes_), 2))
            }
            if self.ordered:
                # Also add in reverse
                __temp_n_class_pairs = self.n_class_pairs_ // 2
                __temp_index_to_pair = {
                    index + __temp_n_class_pairs: (j, i)
                    for index, (i, j)
                    in enumerate(combinations(range(self.n_classes_), 2))
                }
                self.index_to_pair_ = {
                    **self.index_to_pair_,
                    **__temp_index_to_pair
                }

        return self

    @abstractmethod
    def select(self):
        raise NotImplementedError

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_y": True,
        }

    def _get_fit_params(self):
        return ['multiplier']

    def _get_select_params(self):
        return ['coverage', 'max_iters']


class GreedyCoverSelector(SetCoverSelectorBase):
    """Constructs a feature selector based on the greedy cover algorithm.

    Given a data matrix X and a vector of class labels y, select
    a subset of features in X, such that the classes are as `far` from
    each other as possible w.r.t these features.

    E.g.
    ```
    >>> gcs = GreedyCoverSelector(ordered=True)
    >>> gcs.fit(X, y, multiplier=10)
    >>> transformed_X = gcs.transform(X, coverage=10)
    ```

    Parameters
    __________
    ordered: bool, if True will construct the a pairwise matrix
        of ordered pairs
    verbose: bool, whether to show info.
    use_python: bool, if True will switch to a Python implementation
        of the greedy cover algorithm. This is here just for
        debugging purposes as the C++ implementation is much faster.
    operation: callable, operation to use when constructing the class vector.

    Attributes
    __________
    n_samples_: int
        Number of samples seen during ``fit``
    n_features_: int
        Number of features seen during ``fit``
    classes_: ndarray of shape (n_classes,)
        Classes seen in the class label vector during ``fit``
    n_classes_: int
        Number of classes seen in the class label vector during ``fit``
    n_class_pairs_: int
        Number of pairs of classes that get constructed. This
        equals nchoosek(n_classes_, 2) if ordered is False,
        else n_classes_ * (n_classes - 1)
    M_: ndarray of shape (n_class_pairs_, n_features_)
        The pairwise difference matrix
    index, index_to_pair_: dict
        Dictionary that maps row indices in M_ to a pair of
        (ordered) classes
    multiplier: int
        Multiplier used to get a finer resolution when constructing
        multiset multiplicities
    coverage: int or ndarray of shape (n_class_pairs_)
        Coverage requested during ``predict``
    max_iters: int
        Maximum number of features to select during ``predict``
    n_outputs_: int
        Number of features selected during ``predict``
    n_pairs_with_incomplete_cover_: int
        Number of pairs (elements) that could not be covered
        to the desired coverage during ``predict``
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            use_python=False,
            multiplier=None,
            operation=lambda x: x.mean(axis=0)):
        super().__init__(ordered=ordered, verbose=verbose, operation=operation)
        self.use_python = use_python
        self._multiplier = multiplier

    def select(self, *, coverage, max_iters=0):
        """
        Returns the indices of the selected features given a certain coverage.

        Parameters
        __________
        coverage: int or list, in case of a list will apply a specific
            coverage to each element.
        max_iters: maximum number of iterations (features) to return.
            A value of 0 or None means no limit.
        """
        check_is_fitted(self)
        self.multiplier = self._multiplier

        self.coverage = coverage
        self.max_iters = max_iters
        # max_iters == 0 means no limit on the number of iterations
        if max_iters is None or max_iters < 0:
            max_iters = 0

        solution = self._gci_wrapper.predict(coverage, max_iters)
        self.solution_indices_ordered_ = solution
        self.feature_importances_ = np.zeros(self.n_features_)
        self.feature_importances_[solution] = np.arange(
            len(solution) + 1, 1, -1)
        if self.n_pairs_with_incomplete_cover_ > 0 and self.verbose:
            logger.warn("Could not cover "
                        f"{self.n_pairs_with_incomplete_cover_} elements.")

        self.n_outputs_ = len(solution)
        logger.info(f"Selected {self.n_outputs_} multisets.")

        return self

    def plot_progress(self):
        """
        Plots the number of remaining elements to be covered,
        and the coverage reached for every feature selected.
        """
        fig, ax1 = plt.subplots()
        ax1.plot(self._gci_wrapper.n_elements_remaining_)
        ax1.set_ylabel('N remaining elements')
        ax1.set_xlabel('N features')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        coverage_until = self._gci_wrapper.coverage_until_
        ax2.plot(coverage_until, color='red')
        ax2.set_ylabel('Coverage')
        ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        plt.show()

    @property
    def n_elements_remaining_(self):
        return self._gci_wrapper.n_elements_remaining_

    @property
    def coverage_until_(self):
        return self._gci_wrapper.coverage_until_

    def feature_coverage(self, index, effective=False):
        """
        Returns a heatmap of pair coverages for the feature given at index.

        Parameters
        __________
        index: int, feature index in X
        effective: bool, if True will plot the corrected coverage
            of the feature, i.e., after stronger features may have been
            selected
        """
        check_is_fitted(self)

        # length self.n_class_pairs_
        if effective:
            elements, multiplicity = self._gci_wrapper.effective_at(index)
        else:
            elements, multiplicity = self._gci_wrapper[index]
        heatmap = np.zeros((self.n_classes_, self.n_classes_))

        for element, mult in zip(elements, multiplicity):
            i1, i2 = self.index_to_pair_[element]
            heatmap[i1][i2] = mult

        return heatmap

    def max_coverage(self):
        """
        Returns a heatmap of max pair coverages.
        """
        check_is_fitted(self)
        heatmap = np.zeros((self.n_classes_, self.n_classes_))
        max_coverage = self._gci_wrapper.max_coverage_

        for key in self.index_to_pair_:
            p1, p2 = self.index_to_pair_[key]
            heatmap[p1][p2] = max_coverage[key]

        return heatmap

    @staticmethod
    def plot_heatmap(hm, classes=None, logp1=False):
        """
        Plot a heatmap.

        Parameters
        __________
        hm: square matrix representing the heatmap
        classes: None or array of class labels
        """
        plot_heatmap(hm, classes, exclude_diag=True, logp1=logp1)

    @property
    def n_pairs_with_incomplete_cover_(self):
        check_is_fitted(self)
        return len(self._gci_wrapper.elements_incomplete_cover_)

    @property
    def pairs_with_incomplete_cover_(self):
        """
        Returns pairs of classes that were not covered fully by the
        algorithm. Uses the original class labels.
        """
        check_is_fitted(self)
        int_pairs = [self.index_to_pair_[m]
                     for m in self._gci_wrapper.elements_incomplete_cover_]
        return [(self.classes_[i], self.classes_[j]) for i, j in int_pairs]

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, m):
        """
        Triggers a change on GCIWrapper everytime multiplier is changed
        by re-calling ``fit`` and re-constructing the multisets from M.
        """
        if m is not None and m <= 0:
            raise ValueError("Multiplier must be positive.")

        self._multiplier = m
        if self.use_python:
            logger.info("Using Python")
            self._gci_wrapper = GCIPython(self.verbose)
        else:
            self._gci_wrapper = GCIWrapper(
                self.M_.shape[0], self._multiplier, self.verbose)
        self._gci_wrapper.fit(self.M_)


class CEMSelector(SetCoverSelectorBase):
    """Set cover based on the cross entropy method.

    Parameters
    __________
    ordered: bool, if True will construct the a pairwise matrix
        of ordered pairs
    verbose: bool, whether to show info.
    max_iters: int
        Maximum number of iterations for the cross entropy method.
    N: int
        Number of random samples to draw per iteration.
    rho: float in (0, 1)
        quantile to use when selecting gamma_hat.
    alpha: float:
        Parameter that controls the tradeoff between coverage desired
        and the number of features selected. If None, will use
        alpha = coverage * 10 / n_features_
    smoothing_parameter: float
        Parameter to use during exponential smoothing update. If None,
        will not apply smoothing.
    eps: float
        Tolerance to use for stopping the algorithm.
    patience: int
        Number of iterations to wait before stopping the algorithm.
    operation: callable, operation to use when constructing the class vector.
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            max_iters=500,
            rs=1000,
            rho=0.1,
            alpha=None,
            smoothing_parameter=0.7,
            eps=1e-4,
            patience=5,
            operation=lambda x: x.mean(axis=0)):
        super().__init__(ordered=ordered, verbose=verbose, operation=operation)
        self.max_iters = max_iters
        self.rs = rs
        self.rho = rho
        self.alpha = alpha
        self.smoothing_parameter = smoothing_parameter
        self.eps = eps
        self.patience = patience

    def select(self, *, coverage):
        check_is_fitted(self)
        # round
        # self.M_ = np.round(self.M_).astype(int)
        self.pairs_with_incomplete_cover_ = np.argwhere(
            self.M_.sum(axis=1) < coverage).flatten()
        self.n_pairs_with_incomplete_cover_ = self.pairs_with_incomplete_cover_.size
        if self.n_pairs_with_incomplete_cover_ > 0:
            logger.info(
                f"Cannot cover {self.n_pairs_with_incomplete_cover_} elements.")
        self.coverage = coverage
        self.feature_importances_ = np.zeros(self.n_features_)

        # initial probabilities
        v_hat = np.full_like(self.feature_importances_, 1/2)
        no_change_iter = 0

        self.coverages_ = []
        self.average_n_features_selected_ = []
        self.ordered_features_ = []

        for iter in range(self.max_iters):
            # draw random samples
            samples = np.random.binomial(1, v_hat, (self.rs, self.n_features_)).astype(float)
            # compute scores for each sample
            scores = self._score(samples, self.alpha)
            # find quantile
            gamma_hat = np.quantile(scores, 1 - self.rho, interpolation='higher')
            prev_vhat = v_hat.copy()
            # update v-hat
            v_hat = ((scores >= gamma_hat)[:, np.newaxis] * samples).sum(axis=0)
            v_hat /= (scores >= gamma_hat).sum()
            # smoothed update
            if self.smoothing_parameter is not None:
                v_hat = (
                    self.smoothing_parameter * v_hat +
                    (1 - self.smoothing_parameter) * prev_vhat
                )
            # check if converged
            logger.info(v_hat.max())
            logger.info(np.abs(v_hat - prev_vhat).max())
            if np.abs(v_hat - prev_vhat).max() <= self.eps:
                no_change_iter += 1
                if no_change_iter == self.patience:
                    logger.info(f"Converged with eps={self.eps}.")
                    break
            else:
                no_change_iter = 0

            new_added = (
                set(np.argwhere(v_hat > 0.98).flatten().tolist()) -
                set(self.ordered_features_)
            )
            if len(new_added) > 0:
                self.ordered_features_ += list(new_added)
            high_prob_feats = np.argwhere(v_hat > 0.98).flatten()
            logger.info(f"{high_prob_feats.size} high probability features.")
            # logger.info(f"{np.argwhere(v_hat > 0.5).flatten()}")
            per_element_coverage = self.M_[:, high_prob_feats].sum(axis=1)
            logger.info(f"{per_element_coverage.min()} smallest coverage.")

        high_prob_feats = np.argwhere(v_hat > 0.98).flatten()
        assert set(high_prob_feats).issubset(set(self.ordered_features_))
        self.ordered_features_ = [i for i in self.ordered_features_
                                  if i in high_prob_feats]

        self.feature_importances_[self.ordered_features_] = np.arange(
            len(self.ordered_features_) + 1, 1, -1)

        logger.info(f"{high_prob_feats.size} high probability features.")
        per_element_coverage = self.M_[:, high_prob_feats].sum(axis=1)
        self.min_coverage = per_element_coverage.min()
        logger.info(f"{self.min_coverage} smallest coverage.")
        logger.info(f"{per_element_coverage.mean()} average coverage.")
        logger.info(f"{np.percentile(per_element_coverage, 10)} 10 perc coverage.")
        self.coverages_ = np.asarray(self.coverages_)
        self.ordered_features_ - np.asarray(self.ordered_features_)
        self.average_n_features_selected_ = np.asarray(self.average_n_features_selected_)

        return self

    def _score(self, samples, alpha=None):
        """Score function for given random samples.

        samples: array-like of shape (N, n_features)
        alpha: hyperparameter determining trade-off between coverage
            and number of genes picked.

        Returns:
        scores: np.ndarray of shape (N,)
        """
        element_by_sample = self.M_ @ samples.T
        # Clip to desired coverage
        element_by_sample = np.clip(element_by_sample, 0, self.coverage)
        coverage_per_sample = element_by_sample.min(axis=0)
        genes_picked_per_sample = samples.sum(axis=1)
        if alpha is None:
            alpha = self.coverage / self.n_features_
        self.coverages_.append(coverage_per_sample.mean())
        self.average_n_features_selected_.append(genes_picked_per_sample.mean())
        return coverage_per_sample - alpha * genes_picked_per_sample


class FReliefF(FeatureSelector):
    """Implements the Fast ReliefF algorithm that relies on approximate
    neighbors for large datasets.

    Parameters
    __________
    n_neighbors: int, number of neighbors to consider for each class
    algorithm: str, 'auto', 'exact' or 'approx', the algorithm
        to use for computing neighbors.
        'auto' will select 'approx' if number of points is greater
        than 1000.
    n_jobs: int, number of threads to use. Only used when 'exact'
        neighbors mode is selected.
    square: bool, whether to square the difference in scores or
        take abs.
    vertical: 0, 1, or 2, can be faster if node has large memory or number
        of neighbors is small. When 0 will for-loop along samples. If 1,
        will for-loop along n_neighbors.
        If 2, operations are entirely vectorized.
    """

    def __init__(
            self,
            *,
            n_neighbors=1,
            algorithm='auto',
            n_jobs=-1,
            square=True,
            verbose=True,
            vertical=False):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.square = square
        self.op = np.square if square else np.abs
        self.verbose = verbose
        self.vertical = vertical

    @property
    def is_auto(self):
        return self.algorithm == 'auto'

    def fit(self, X, y):
        """
        Determine feature scores based on the data X and class labels y.
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )

        if self.is_auto:
            algo = 'approx' if X.shape[0] > 5_000 else 'exact'
        else:
            algo = self.algorithm

        self.n_samples_, self.n_features_ = X.shape
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        # Class fractions
        self.priors_ = counts / self.n_samples_
        # Feature scores
        scores = np.zeros(self.n_features_)

        if self.verbose:
            to_enum = tqdm(zip(self.classes_, self.priors_), total=len(self.classes_))
        else:
            to_enum = zip(self.classes_, self.priors_)

        for label, prior in to_enum:
            positive_idx = np.argwhere(y == label).flatten()
            negative_idx = np.argwhere(y != label).flatten()
            gc.collect()

            # Fit only positive points
            # Add 1 neighbor to count for `self`
            neighs_to_use = min(self.n_neighbors + 1, positive_idx.size)
            FN = FastNeighbor(neighs_to_use, algo, self.n_jobs)
            FN.fit(X[positive_idx])
            # Neighbors from the positive class for all points
            indices = FN.predict(X)  # shape (n_samples, neighs_to_use)
            # convert to the correct indices
            indices = positive_idx[indices]
            # Need to separate score updates depending on whether
            # a point is positive or negative
            assert indices.shape == (self.n_samples_, neighs_to_use)
            c_indices = np.zeros((self.n_samples_, neighs_to_use - 1), dtype=int)
            # For negative samples, there is no risk of including `self`
            # so we simply take the first neighs_to_use - 1
            c_indices[negative_idx] = indices[negative_idx][:, :-1]
            # For positive samples, take the last neighs_to_use - 1
            # since the first one is `self`
            c_indices[positive_idx] = indices[positive_idx][:, 1:]

            # Update scores
            if self.vertical == 2:
                # Fastest if a lot of memory is available, otherwise slow
                pairwise_diff = X[:, None, :] - X[c_indices]
                squared_means = self.op(pairwise_diff).mean(axis=1)
                scores -= squared_means[positive_idx].sum(axis=0)
                scores += prior * squared_means[negative_idx].sum(axis=0)
            elif self.vertical == 1:
                # Fast if a lot of memory is available, otherwise slow
                _temp_diffs = np.zeros_like(X)
                for _i in range(c_indices.shape[1]):
                    _temp_diffs += self.op(X - X[c_indices[:, _i]]) / c_indices.shape[1]
                scores -= _temp_diffs[positive_idx].sum(axis=0)
                scores += prior * _temp_diffs[negative_idx].sum(axis=0)
            else:
                for i in range(self.n_samples_):
                    neigh_indices = c_indices[i]  # array of size neighs_to_use
                    cur_scores = (self.op(X[i] - X[neigh_indices])).mean(axis=0)
                    if y[i] == label:
                        scores -= cur_scores
                    else:
                        scores += prior * cur_scores

        scores /= self.n_samples_
        self.feature_importances_ = scores

        return self

    def select(self):
        return self

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_y": True,
        }


class TopDE(FeatureSelector):
    def __init__(self, is_logged=True, verbose=True):
        """
        Find the top DE features for every class.
        """
        self.is_logged = is_logged
        self.verbose = verbose

    def fit(self, X, y, gene_names=None):
        """
        For every class label in y, compute differentially expressed genes
        by using a t-test. The genes will be ranked by fold change.
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=True,
            ensure_min_samples=2,
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )

        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.de_genes_ = {}

        for class_ in tqdm(self.classes_):
            grouping = np.zeros(self.n_samples_)
            grouping[y == class_] = 1
            test = de.test.t_test(
                data=X,
                grouping=grouping,
                gene_names=gene_names if gene_names is not None
                else np.arange(self.n_features_).astype(str),
                is_logged=self.is_logged
            ).summary()

            test = test.sort_values(by='log2fc', ascending=False)
            self.de_genes_[class_] = test.copy()

    def dump(self, folderpath):
        """Dump DE results into a folder.
        """
        check_is_fitted(self)
        if not os.path.isdir(folderpath):
            os.mkdir(folderpath)
        for class_ in self.de_genes_:
            self.de_genes_[class_].to_csv(
                os.path.join(folderpath, f'{class_}.csv'))

    def load(self, folderpath):
        """Load DE results from a folder.
        """
        files = os.listdir(folderpath)
        self.de_genes_ = {}

        for file in files:
            if not file.endswith('csv'):
                continue
            test = pd.read_csv(os.path.join(folderpath, file), index_col=0)
            self.de_genes_[file[:-4]] = test

        self.n_classes_ = len(self.de_genes_)
        self.classes_ = np.array(list(self.de_genes_.keys()))
        self.n_features_ = len(self.de_genes_[self.classes_[0]])

    def select(self, n_feats_per_class):
        check_is_fitted(self)
        if self.verbose:
            logger.info(f"Trying {n_feats_per_class} features per class for"
                        f" a total of {n_feats_per_class * self.n_classes_} features.")

        features = []
        for class_ in self.classes_:
            sorted_idx = self.de_genes_[class_].index.to_numpy()
            features.append(sorted_idx[:n_feats_per_class])
        features = np.unique(np.concatenate(features))

        if self.verbose:
            logger.info(f"Ultimately selected {len(features)} features.")

        self.n_outputs_ = len(features)
        self.feature_importances_ = np.zeros(self.n_features_)
        self.feature_importances_[features] = 1
        return self

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_y": True,
        }

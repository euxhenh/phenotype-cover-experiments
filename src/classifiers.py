import json
import os

import jsbeautifier
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression

from ._logger import logger
from .utils.plots import plot_heatmap


class Classifier(BaseEstimator):
    """
    Wrapper for a logisitc regression classifier.
    """

    def __init__(
            self,
            model,
            *,
            feature_selector=None,
            features=None,
            scale=False,
            normalize=False,
            **kwargs):
        """
        Initializes an classifier model.

        Args:
            model: an object that implements ``fit`` and ``predict``.
            feature_selector: None or an object
                that implements a transform method.
            features: None or ndarray
                If not None, must contain indices of features in X.
                Will restrict training to X[:, features] only.
                This will be ignored if feature_selector is not None.
            scale: bool, whether to scale every feature to zero mean
                and unit variance
            normalize: bool, whether to scale the data to 0-1.

        Attributes:
            n_samples_: int, number of samples seen during ``fit``
            n_features_in_: int, number of features seen during ``fit``
            n_features_: int, number of features actually used for fitting
                if features is not None
            classes_: ndarray, uniqu e class labels seen during ``fit``
        """
        super().__init__()
        self.feature_selector = feature_selector
        self.features = features
        self.scale = scale
        self.normalize = normalize

        self._model = model

    def fit(self, X, y):
        """
        Fits the model with data matrix X and class label vector y.

        Args:
            X: ndarray, data matrix of shape (n_samples, n_features)
            y: ndarray, class labels of shape (n_samples,)

        """
        if X.ndim > 2:
            raise ValueError("Data matrix must be two dimensional.")
        if y.ndim > 1:
            raise ValueError("Class label vector must be one dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples must equal number of labels.")

        self.stats = {}

        self.n_samples_, self.n_features_in_ = X.shape
        self.n_features_ = self.n_features_in_

        self.stats['n_train'] = self.n_samples_
        self.stats['n_features_in'] = self.n_features_in_

        # Perform any feature selection
        X = self._select_features(X)
        if sp.issparse(X):
            X = X.toarray()
        self.n_features_ = X.shape[1]
        self.stats['n_features'] = self.n_features_
        logger.info(f"Using {self.n_features_} features.")
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.stats['n_classes'] = self.n_classes_

        preps = []
        if self.scale:
            preps.append(StandardScaler())
        if self.normalize:
            preps.append(Normalizer())
        self.stats['scale'] = self.scale
        self.stats['normalize'] = self.normalize

        if len(preps) > 0:
            self._pipe = make_pipeline(*preps, self._model)
        else:
            self._pipe = make_pipeline(self._model)

        self._pipe.fit(X, y)

    def predict(self, X):
        """
        Given a data matrix X, predict class labels for every sample.
        """
        X = self._select_features(X)
        if sp.issparse(X):
            X = X.toarray()
        return self._pipe.predict(X)

    def confuse(self, X, y, preds=None):
        """
        Returns a confusion matrix for the predictions of X and gt labels y.
        """
        if preds is None:
            preds = self.predict(X)
        cm = confusion_matrix(y, preds, labels=self.classes_)
        return cm

    @staticmethod
    def plot_heatmap(hm, classes=None, savepath=None, logp1=True):
        """
        Plot a heatmap.

        Args:
            hm: square matrix representing the heatmap
            classes: None or array of class labels
        """
        plot_heatmap(
            hm, classes, exclude_diag=False, savepath=savepath, logp1=logp1)

    def fit_predict(self, X, y, features=None, scale=False):
        """
        Fits the model with data matrix X and class label vector y
        and returns a prediction vector.

        Args:
            X: ndarray, data matrix of shape (n_samples, n_features)
            y: ndarray, class labels of shape (n_samples,)
            feature_selector: None or an object
                that implements a transform method.
            features: None or ndarray
                If not None, must contain indices of features in X.
                Will restrict training to X[:, features] only.
                This will be ignored if feature_selector is not None.
            scale: bool, whether to scale every feature to zero mean
                and unit variance
        """
        self.fit(X, y, features, scale)
        return self.predict(X)

    def score(self, X, y):
        """
        Predicts class labels using X and computes the accuracy
        between the predictions and y.
        """
        check_is_fitted(self)
        X = self._select_features(X)
        if sp.issparse(X):
            X = X.toarray()
        print(f'Accuracy: {self._pipe.score(X, y):.4f}')

    def report(self, X, y, ret=False):
        """
        Predicts class labels using X and computes several classification
        metric between the predictions and y.
        """
        check_is_fitted(self)
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        f1micro = f1_score(y, preds, average='micro')
        f1macro = f1_score(y, preds, average='macro')
        f1weighted = f1_score(y, preds, average='weighted')
        # Need to select features for predict_proba
        X = self._select_features(X)
        if sp.issparse(X):
            X = X.toarray()

        roc = 0
        if isinstance(self._model, LogisticRegression):
            roc = roc_auc_score(
                y, self._pipe.predict_proba(X), multi_class='ovr')
        if ret:
            return acc, f1micro, f1macro, f1weighted, roc
        else:
            print("Accuracy:", acc)
            print("F1 micro:", f1micro)
            print("F1 macro:", f1macro)
            print("F1 weighted:", f1weighted)
            print("ROC AUC:", roc)

    def dump(
            self, x_train, y_train, x_test, y_test, *,
            json_path, confusion_matrix_dir, key, extras):
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                dic = json.load(f)
        else:
            dic = {}

        acc, f1micro, f1macro, f1weighted, roc = self.report(x_train, y_train, ret=True)
        self.stats['train_accuracy'] = acc
        self.stats['train_f1_micro'] = f1micro
        self.stats['train_f1_macro'] = f1macro
        self.stats['train_f1_weighted'] = f1weighted
        self.stats['train_ROC'] = roc

        acc, f1micro, f1macro, f1weighted, roc = self.report(x_test, y_test, ret=True)
        self.stats['n_test'] = x_test.shape[0]
        self.stats['test_accuracy'] = acc
        self.stats['test_f1_micro'] = f1micro
        self.stats['test_f1_macro'] = f1macro
        self.stats['test_f1_weighted'] = f1weighted
        self.stats['test_ROC'] = roc

        if not os.path.isdir(confusion_matrix_dir):
            os.mkdir(confusion_matrix_dir)
        cm = self.confuse(x_train, y_train)
        self.plot_heatmap(cm, self.classes_, savepath=os.path.join(
            confusion_matrix_dir, 'train-' + str(key) + '.pdf'))
        cm = self.confuse(x_test, y_test)
        self.plot_heatmap(cm, self.classes_, savepath=os.path.join(
            confusion_matrix_dir, 'test-' + str(key) + '.pdf'))

        for kk in extras:
            if isinstance(extras[kk], np.ndarray):
                self.stats[kk] = extras[kk].tolist()
            else:
                self.stats[kk] = extras[kk]

        dic[key] = self.stats

        options = jsbeautifier.default_options()
        options.indent_size = 4
        beau_report = jsbeautifier.beautify(json.dumps(dic), options)
        with open(json_path, "w") as f:
            f.write(beau_report)

    def _select_features(self, X):
        """
        If any of self.feature_selector or self.features is not None,
        will use these to select features from X and return it.
        """
        if self.feature_selector is not None:
            if not hasattr(self.feature_selector, 'transform'):
                raise KeyError("Feature selector object must implement "
                               "a ``transform`` method.")
            X = self.feature_selector.transform(X)
        elif self.features is not None:
            X = X[:, self.features]
        else:
            return X

        return X

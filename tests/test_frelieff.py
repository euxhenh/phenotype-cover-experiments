import unittest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.feature_selection import SelectFromModel

from src.feature_selector import FReliefF


class Test(unittest.TestCase):
    def test1(self):
        n_samples = 200
        n_features = 40
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 5, n_samples)

        scores = np.zeros(n_features)

        unq_y, counts = np.unique(y, return_counts=True)
        priors = counts / n_samples

        for i in range(n_samples):
            x = X[i].copy()
            label = y[i].copy()

            xdel = np.delete(X, i, axis=0)  # remove self
            ydel = np.delete(y, i)

            nh_x = np.argmin((np.square(x - xdel[ydel == label])).sum(axis=1))
            scores -= np.square(x - xdel[ydel == label][nh_x])

            for yx, prior in zip(unq_y, priors):
                if yx == label:
                    continue
                nm_x = np.argmin((np.square(x - X[y == yx])).sum(axis=1))
                scores += prior * np.square(x - X[y == yx][nm_x])

        scores /= n_samples
        frf = FReliefF(n_neighbors=1, algorithm='exact')
        frf.fit(X, y)
        assert_allclose(scores, frf.get_scores())

        frf = FReliefF(n_neighbors=1, algorithm='approx')
        frf.fit(X, y)
        assert_allclose(scores, frf.get_scores(), rtol=0.5, atol=0.001)

    def test_mnist(self):
        from mnist import MNIST
        import matplotlib.pyplot as plt

        mndata = MNIST('/Users/ehasanaj/code/python/relief/data')
        images, labels = mndata.load_training()
        images, labels = np.array(images), np.array(labels).astype(int)

        images = images / 255
        rr = FReliefF(n_neighbors=10)
        print(images.shape, labels.shape)

        rr.fit(images, labels)
        plt.imshow(rr.get_scores().reshape(28, 28))
        plt.show()

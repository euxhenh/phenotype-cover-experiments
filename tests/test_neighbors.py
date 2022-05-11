import unittest
import numpy as np
from numpy.testing import assert_allclose

from src.neighbors import FastNeighbor


class Test(unittest.TestCase):
    def test1(self):
        fn = FastNeighbor(1, algorithm='approx')

        X = np.array([
            [0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [2, 5, 2, 7, 1],
            [6, 5, 2, 7, 1],
            [2, 5, 2, 7, 2]
        ])

        fn.fit(X)

        nn = fn.predict()
        assert_allclose(nn, np.array([[1, 0, 4, 2, 2]]).T)
        nn = fn.predict(X)
        assert_allclose(nn, np.arange(5).reshape(-1, 1))

    def test2(self):
        fn = FastNeighbor(1, algorithm='exact')

        X = np.array([
            [0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [2, 5, 2, 7, 1],
            [6, 5, 2, 7, 1],
            [2, 5, 2, 7, 2]
        ])

        fn.fit(X)

        nn = fn.predict()
        assert_allclose(nn, np.array([[1, 0, 4, 2, 2]]).T)
        nn = fn.predict(X)
        assert_allclose(nn, np.arange(5).reshape(-1, 1))

    def test3(self):
        N = 5
        fn = FastNeighbor(N, algorithm='exact')

        X = np.random.random((20, 40))
        fn.fit(X)

        nn = fn.predict()
        for i in range(X.shape[0]):
            dists = ((X - X[i])**2).sum(axis=1)
            nneighs = np.argsort(dists)[1:N+1]
            assert_allclose(nn[i], nneighs)

        nn = fn.predict(X)
        for i in range(X.shape[0]):
            dists = ((X - X[i])**2).sum(axis=1)
            nneighs = np.argsort(dists)[:N]
            assert_allclose(nn[i], nneighs)

    def test4(self):
        N = 5
        fn = FastNeighbor(N, algorithm='approx')

        X = np.random.random((20, 40))
        fn.fit(X)

        nn = fn.predict()
        for i in range(X.shape[0]):
            dists = ((X - X[i])**2).sum(axis=1)
            nneighs = np.argsort(dists)[1:N+1]
            assert_allclose(nn[i], nneighs)

        nn = fn.predict(X)
        for i in range(X.shape[0]):
            dists = ((X - X[i])**2).sum(axis=1)
            nneighs = np.argsort(dists)[:N]
            assert_allclose(nn[i], nneighs)

import unittest
import numpy as np
from numpy.testing import assert_allclose

from src.gci_wrapper import GCIPython, GCIWrapper

def wrap(gci, x):
    gci.fit(x)

    assert gci.n_elements == 4
    assert gci.n_multisets_ == 5

    assert_allclose(gci.max_coverage_, [14, 8, 7, 2])

    solution = gci.predict(2)
    assert_allclose(solution, [3, 1, 4])
    n_elements_rem = gci.n_elements_remaining_
    assert_allclose(n_elements_rem, [2, 1, 0])
    coverage_until = gci.coverage_until_
    assert_allclose(coverage_until, [0, 1, 2])

    solution = gci.predict(3)
    assert_allclose(solution, [3, 4, 1])
    elements_incomplete_cover_ = gci.elements_incomplete_cover_
    assert_allclose(elements_incomplete_cover_, [3])
    coverage_until = gci.coverage_until_
    assert_allclose(coverage_until, [1, 2, 3])

x = np.array([
    [2, 1, 5, 6, 0],
    [1, 3, 1, 3, 0],
    [0, 1, 3, 1, 2],
    [0, 1, 0, 0, 1]
])

class TestSelectors(unittest.TestCase):
    def test1(self):
        wrap(GCIPython(), x)

    def test2(self):
        wrap(GCIWrapper(x.shape[0]), x)

    def test3(self):
        N = 1000
        xx = np.random.randint(0, 10, (N, 5000))
        gcip = GCIPython()
        gciw = GCIWrapper(N)

        gcip.fit(xx)
        gciw.fit(xx)

        assert_allclose(gcip.max_coverage_, gciw.max_coverage_)
        assert_allclose(gcip.predict(5), gciw.predict(5))
        assert_allclose(gcip.n_elements_remaining_, gciw.n_elements_remaining_)
        assert_allclose(gcip.coverage_until_, gciw.coverage_until_)
        assert_allclose(gcip.elements_incomplete_cover_, gciw.elements_incomplete_cover_)

        assert_allclose(gcip.predict(10), gciw.predict(10))
        assert_allclose(gcip.n_elements_remaining_, gciw.n_elements_remaining_)
        assert_allclose(gcip.coverage_until_, gciw.coverage_until_)
        assert_allclose(gcip.elements_incomplete_cover_, gciw.elements_incomplete_cover_)

        assert_allclose(gcip.predict(1000), gciw.predict(1000))
        assert_allclose(gcip.n_elements_remaining_, gciw.n_elements_remaining_)
        assert_allclose(gcip.coverage_until_, gciw.coverage_until_)
        assert_allclose(gcip.elements_incomplete_cover_, gciw.elements_incomplete_cover_)
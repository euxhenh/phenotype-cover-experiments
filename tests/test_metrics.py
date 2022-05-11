import unittest

from numpy.testing import assert_allclose

from src.metrics import stability, iou


class stability_test(unittest.TestCase):
    def test_iou(self):
        a = [1, 2, 3, 5, 7]
        b = [1, 3, 6, 7, 9, 0]
        assert_allclose(3 / 8, iou(a, b))

    def test_stability(self):
        sets = [
            [1, 2, 3],
            [2, 3, 4],
            [4, 6, 7]
        ]

        assert_allclose(stability(sets), 1.4/6)

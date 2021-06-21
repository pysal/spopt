import libpysal
import numpy
import unittest

from spopt.region import RegionKMeansHeuristic


RANDOM_STATE = 12345


class TestSkater(unittest.TestCase):
    def setUp(self):

        self.dim = 3
        self.w = libpysal.weights.lat2W(self.dim, self.dim)
        self.data = numpy.random.normal(size=(self.w.n, self.dim))
        self.known_labels = [0, 1, 1, 1, 0, 0, 2, 2, 2]

    def test_region_k_means_heuristic(self):
        numpy.random.seed(RANDOM_STATE)
        model = RegionKMeansHeuristic(self.data, self.dim, self.w)
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.known_labels)

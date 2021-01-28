import numpy
import libpysal
import os
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from spopt.region import RegionKMeansHeuristic


def test_RegionKMeansHeuristic():
    numpy.random.seed(12345)
    w = libpysal.weights.lat2W(3, 3)
    data = numpy.random.normal(size=(w.n, 3))
    RKM = RegionKMeansHeuristic
    model = RKM(data, 3, w)
    model.solve()
    numpy.array_equal(model.labels_, numpy.array([1, 2, 2, 1, 2, 0, 1, 1, 0]))

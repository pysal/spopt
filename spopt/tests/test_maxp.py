import numpy
import libpysal
import geopandas as gpd
import os
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from spopt.region import MaxPHeuristic


def test_MaxPHeuristic():

    pth = libpysal.examples.get_path("mexicojoin.shp")
    mexico = gpd.read_file(pth)
    mexico["count"] = 1
    w = libpysal.weights.Queen.from_dataframe(mexico)
    attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
    threshold = 4
    top_n = 2
    threshold_name = "count"
    numpy.random.seed(123456)
    model = MaxPHeuristic(mexico, w, attrs_name, threshold_name, threshold, top_n)
    model.solve()
    labels = [
        5,
        5,
        3,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
        8,
        6,
        8,
        2,
        2,
        8,
        2,
        8,
        7,
        7,
        2,
        7,
        5,
        3,
        3,
        5,
        3,
        4,
        4,
        4,
        4,
        7,
    ]
    numpy.array_equal(model.labels_, labels)

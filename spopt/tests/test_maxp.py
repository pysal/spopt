import numpy
import libpysal
import geopandas as gpd
import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from .. import MaxPHeuristic


def test_MaxPHeuristic():
    numpy.random.seed(12345)
    gdf = gpd.read_file('../data/n100.shp')
    w = libpysal.weights.Rook.from_dataframe(gdf)
    attrs_name = ['SAR1']
    threshold_name = 'Uniform2'
    threshold = 100
    top_n = 2
    model = MaxPHeuristic(gdf, w, attrs_name, threshold_name, threshold, top_n)
    model.solve()
    numpy.array_equal(model.labels_, numpy.array([1, 2, 2, 1, 2, 0, 1, 1, 0]))


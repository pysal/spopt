
import numpy
import libpysal
import geopandas as gpd
import os
import sys

from spopt.region import RandomRegion, RandomRegions

mdf = gpd.read_file(libpysal.examples.get_path('mexicojoin.shp'))
w = libpysal.weights.Queen.from_dataframe(mdf)
cards = mdf.groupby(by='HANSON03').count().NAME.values.tolist()
ids = mdf.index.values.tolist()

def test_RandomRegion():

    #mdf = gpd.read_file(libpysal.examples.get_path('mexicojoin.shp'))
    #w = libpysal.weights.Queen.from_dataframe(mdf)
    #cards = mdf.groupby(by='HANSON03').count().NAME.values.tolist()
    #ids = mdf.index.values.tolist()
    numpy.random.seed(12345)
    rrmx = RandomRegion(ids, num_regions=6, cardinality = cards)
    regions = [[27, 12, 18, 3, 15, 8],
               [0, 25, 21, 20, 7, 6, 24],
               [23, 10, 13, 11, 19, 16, 26, 14, 17, 22],
               [28, 31], [30, 9, 4], [1, 29, 5, 2]]
    numpy.array_equal(regions, rrmx.regions)

def test_RandomRegions():
    numpy.random.seed(12345)
    rrmxc = RandomRegions(ids, num_regions=6, cardinality = cards,
                         permutations=99)
    regions = [[8, 5, 29, 22, 16, 10],
               [4, 26, 1, 2, 6, 13, 15],
               [19, 3, 20, 11, 31, 12, 0, 17, 7, 21],
               [23, 14],
               [25, 27, 28],
               [18, 9, 30, 24]]
    numpy.array_equal(regions, rrmxc.solutions_feas[2].regions)

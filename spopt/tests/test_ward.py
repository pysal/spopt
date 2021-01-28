import numpy as np
import libpysal
import geopandas as gpd
import os
import sys
from spopt.region import WardSpatial
import unittest


class TestWard(unittest.TestCase):


    def setUp(self):
        self.pth = libpysal.examples.get_path("mexicojoin.shp")
        self.mexico = gpd.read_file(self.pth)
        self.mexico["count"] = 1
        self.attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        np.random.seed(123456)
        self.model = WardSpatial(gdf=self.mexico, w=self.w, attrs_name=self.attrs_name)
        self.model.solve()


    def test_labels_(self):
        observed_labels = self.model.labels_
        known_labels = np.array(
            [
                2,
                2,
                0,
                0,
                0,
                0,
                0,                
                0,
                0,
                0,
                3,
                0,
                0,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                2,
                2,
                2,
                0,
                0,
                0,
                0,
                2,
                2,
                0,
            ]
        )
        np.testing.assert_equal(observed_labels, known_labels)

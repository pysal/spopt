import numpy as np
import libpysal
import geopandas as gpd
from spopt.region import Skater
import unittest

class TestSkater(unittest.TestCase):

    def setUp(self):
        self.pth = libpysal.examples.get_path("mexicojoin.shp")
        self.mexico = gpd.read_file(self.pth)
        self.mexico["count"] = 1
        self.attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        np.random.seed(123456)
        self.model = Skater(gdf=self.mexico, w=self.w, attrs_name=self.attrs_name)
        self.model.solve()

    def test_labels_(self):
        observed_labels = self.model.labels_
        known_labels = np.array(
            [
                0,
                0,
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                3,
                2,
                1,
                1,
                1,
                1,
                4,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
            ]
        )
        np.testing.assert_equal(observed_labels, known_labels)

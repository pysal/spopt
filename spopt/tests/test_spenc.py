import numpy as np
import libpysal
import geopandas as gpd
from .. import Spenc
import unittest


class TestSpenc(unittest.TestCase):


    def setUp(self):
        self.pth = libpysal.examples.get_path("mexicojoin.shp")
        self.mexico = gpd.read_file(self.pth)
        self.mexico["count"] = 1
        self.attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        self.model = Spenc(gdf=self.mexico, w=self.w, attrs_name=self.attrs_name, random_state=123456)
        self.model.solve()


    def test_labels_(self):
        observed_labels = self.model.labels_
        known_labels = np.array(
            [
                4,
                0,
                3,
                1,
                3,
                3,
                3,                
                1,
                2,
                4,
                0,
                1,
                4,
                4,
                0,
                1,
                4,
                4,
                4,
                2,
                3,
                0,
                0,
                1,
                0,
                1,
                3,
                1,
                3,
                1,
                2,
                2,
            ]
        )
        np.testing.assert_equal(observed_labels, known_labels)

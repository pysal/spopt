import numpy as np
import libpysal
import geopandas as gpd
from spopt.region import Spenc
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
        np.testing.assert_equal(len(np.unique(observed_labels)), 5)

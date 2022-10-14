import geopandas
import libpysal
import numpy
import pytest

from spopt.region import WardSpatial


# Empirical tests -- Mexican states
RANDOM_STATE = 12345
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestWard:
    def setup_method(self):

        self.mexico = MEXICO.copy()
        self.mexico["count"] = 1
        self.attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        self.known_labels = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 4, 0]
        self.known_labels += [1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0]

    def test_ward_defaults(self):
        numpy.random.seed(RANDOM_STATE)
        model = WardSpatial(gdf=self.mexico, w=self.w, attrs_name=self.attrs_name)
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.known_labels)

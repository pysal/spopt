import geopandas
import libpysal
import numpy
import pytest

from spopt.region import Spenc


# Empirical tests
RANDOM_STATE = 123456
# -- Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestSpenc:
    def setup_method(self):

        # Mexico
        self.mexico = MEXICO.copy()
        self.w_mexico = libpysal.weights.Queen.from_dataframe(self.mexico)
        self.default_attrs_mexico = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.non_default_mexico = [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2]
        self.non_default_mexico += [1, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]

    def test_spenc_non_defaults(self):
        self.mexico["count"] = 1
        numpy.random.seed(RANDOM_STATE)
        model = Spenc(
            self.mexico,
            self.w_mexico,
            self.default_attrs_mexico,
            random_state=RANDOM_STATE,
            gamma=0,
            n_clusters=3,
        )
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.non_default_mexico)

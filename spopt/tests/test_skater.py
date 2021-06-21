import geopandas
import libpysal
import numpy
from sklearn.metrics import pairwise as skm
import unittest

from spopt.region import Skater


# Empirical tests -- Mexican states
RANDOM_STATE = 12345
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestSkater(unittest.TestCase):
    def setUp(self):
        self.mexico = MEXICO.copy()
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        self.default_attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.default_labels = [0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1]
        self.default_labels += [4, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]

        self.non_default_labels = [0, 0, 1, 2, 2, 3, 4, 5, 6, 4, 4, 2, 5, 7, 7, 5, 7]
        self.non_default_labels += [5, 6, 6, 6, 6, 0, 8, 8, 1, 1, 3, 3, 8, 3, 3]

    def test_skater_defaults(self):
        self.mexico["count"] = 1
        numpy.random.seed(RANDOM_STATE)
        model = Skater(gdf=self.mexico, w=self.w, attrs_name=self.default_attrs_name)
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.default_labels)

    def test_skater_defaults_verbose(self):
        self.mexico["count"] = 1
        sfkws = {"verbose": 1}
        numpy.random.seed(RANDOM_STATE)
        model = Skater(
            self.mexico, self.w, self.default_attrs_name, spanning_forest_kwds=sfkws,
        )
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.default_labels)

    def test_skater_defaults_super_verbose(self):
        self.mexico["count"] = 1
        sfkws = {"verbose": 2}
        numpy.random.seed(RANDOM_STATE)
        model = Skater(
            self.mexico, self.w, self.default_attrs_name, spanning_forest_kwds=sfkws,
        )
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.default_labels)

    def test_skater_defaults_non_defaults(self):
        self.mexico["count"] = 1
        args = self.mexico, self.w, self.default_attrs_name
        n_clusters, floor, trace, islands = 10, 3, True, "ignore"
        kws = dict(n_clusters=n_clusters, floor=floor, trace=trace, islands=islands)
        sfkws = dict(
            dissimilarity=skm.manhattan_distances,
            affinity=None,
            reduction=numpy.sum,
            center=numpy.mean,
            verbose=2,
        )
        kws.update(dict(spanning_forest_kwds=sfkws))
        numpy.random.seed(RANDOM_STATE)
        model = Skater(*args, **kws)
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.non_default_labels)

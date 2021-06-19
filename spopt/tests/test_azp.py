import libpysal
import geopandas
import numpy
import unittest

from spopt.region import AZP


# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestAZP(unittest.TestCase):
    def setUp(self):

        self.mexico = MEXICO.copy()

        # labels for from_w:
        # count=1, n_clusters=5
        self.basic_from_w_labels = [0, 0, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2]
        self.basic_from_w_labels += [1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1]

        ## labels for:
        ## count=4, threshold=10, top_n=5
        # self.complex_labels = [3, 3, 2, 10, 10, 2, 6, 9, 1, 6, 6, 10, 7, 4, 4, 7]
        # self.complex_labels += [4, 7, 1, 1, 9, 1, 3, 8, 8, 2, 8, 2, 5, 5, 5, 9]

        ## labels for one variable column:
        ## count=1, threshold=5, top_n=5
        # self.var1_labels = [3, 3, 6, 2, 2, 2, 2, 4, 2, 4, 5, 2, 5, 1, 1, 5, 1]
        # self.var1_labels += [4, 5, 5, 1, 1, 3, 3, 6, 3, 6, 4, 4, 6, 6, 4]

    def test_azp_basic_from_w(self):
        self.mexico["count"] = 1
        w = libpysal.weights.Queen.from_dataframe(self.mexico)
        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        args = (self.mexico, w, attrs_name)
        model = AZP(*args, n_clusters=3, random_state=123456)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.basic_from_w_labels)

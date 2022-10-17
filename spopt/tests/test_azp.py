import libpysal
import geopandas
import numpy
import pytest

import spopt
from spopt.region import AZP


RANDOM_STATE = 123456

# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestAZP:
    def setup_method(self):

        self.mexico = MEXICO.copy()

        # labels for from_w:
        # n_clusters=3, basic AZP
        self.basic_from_w_labels = [0, 0, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2]
        self.basic_from_w_labels += [1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1]

        # labels for:
        # n_clusters=3, simulated annealing AZP variant
        # self.simann_from_w_labels = [0, 0, 0, 0, 0, 1, 2, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1]
        # self.simann_from_w_labels += [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2]

    def test_azp_basic_from_w(self):
        w = libpysal.weights.Queen.from_dataframe(self.mexico)
        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        args = (self.mexico, w, attrs_name)
        kwargs = {"n_clusters": 3, "random_state": RANDOM_STATE}
        model = AZP(*args, **kwargs)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.basic_from_w_labels)

    # def test_azp_sim_anneal_from_w(self):
    #    w = libpysal.weights.Queen.from_dataframe(self.mexico)
    #    attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
    #    sim_ann = spopt.region.azp_util.AllowMoveAZPSimulatedAnnealing(
    #        10, sa_moves_term=10
    #    )
    #    args = (self.mexico, w, attrs_name)
    #    kwargs = {
    #        "n_clusters": 3,
    #        "random_state": RANDOM_STATE,
    #        "allow_move_strategy": sim_ann,
    #    }
    #    model = AZP(*args, **kwargs)
    #    model.solve()
    #
    #    numpy.testing.assert_array_equal(model.labels_, self.simann_from_w_labels)

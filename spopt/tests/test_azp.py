import geopandas
import libpysal
import numpy
from packaging.version import Version

from spopt.region import AZP

# see gh:spopt#437
LIBPYSAL_GE_48 = Version(libpysal.__version__) >= Version("4.8.0")
w_kwargs = {"use_index": True} if LIBPYSAL_GE_48 else {}


RANDOM_STATE = 123456

# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestAZP:
    def setup_method(self):
        self.mexico = MEXICO.copy()

        # labels for from_w:
        # n_clusters=3, basic AZP
        self.basic_from_w_labels = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 2]
        self.basic_from_w_labels += [1, 1, 1, 2, 1, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2]

        # labels for:
        # n_clusters=3, simulated annealing AZP variant
        # self.simann_from_w_labels = [
        #    0, 0, 0, 0, 0, 1, 2, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1
        # ]
        # self.simann_from_w_labels += [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2]

    def test_azp_basic_from_w(self):
        w = libpysal.weights.Queen.from_dataframe(self.mexico, **w_kwargs)

        weights = {}
        for k, v in w.neighbors.items():
            arr = []
            x = self.mexico.iloc[k].geometry.centroid
            for n in v:
                y = self.mexico.iloc[n].geometry.centroid
                arr.append(x.distance(y))
            weights[k] = arr
        neighbors = w.neighbors
        w = libpysal.weights.W(neighbors, weights)

        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        args = (self.mexico, w, attrs_name)
        kwargs = {"n_clusters": 3, "random_state": RANDOM_STATE}
        model = AZP(*args, **kwargs)
        model.solve()

        # print(model.labels_)

        numpy.testing.assert_array_equal(model.labels_, self.basic_from_w_labels)

    # def test_azp_sim_anneal_from_w(self):
    #    w = libpysal.weights.Queen.from_dataframe(self.mexico, **w_kwargs)
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

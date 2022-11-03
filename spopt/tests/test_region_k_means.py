import libpysal
import numpy
import pandas
import pytest

from spopt.region import RegionKMeansHeuristic
from spopt.region.spenclib.utils import lattice


RANDOM_STATE = 12345


class TestRegionKMeansHeuristic:
    def setup_method(self):

        # small 3 x 3 example w/ 3 regions
        dim_small = 3
        self.w_small = libpysal.weights.lat2W(dim_small, dim_small)
        numpy.random.seed(RANDOM_STATE)
        self.data_small = numpy.random.normal(size=(self.w_small.n, dim_small))
        self.reg_small = 3
        self.known_labels_small = [1, 1, 1, 0, 1, 2, 0, 0, 2]

        # large 20 x 20 example broken into 2 islands w/ 5 regions
        hori, vert = 20, 20
        n_polys = hori * vert
        gdf = lattice(hori, vert)
        numpy.random.seed(RANDOM_STATE)
        gdf["data_values_1"] = numpy.random.random(n_polys)
        gdf["data_values_2"] = numpy.random.random(n_polys)
        gdf = pandas.concat([gdf[:200], gdf[220:]])
        self.w_large = libpysal.weights.Rook.from_dataframe(gdf)
        self.data_large = gdf[["data_values_1", "data_values_2"]].values
        self.reg_large = 3
        self.limit_index = 30
        self.known_labels_large = [1] * self.limit_index

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_region_k_means_heuristic_synth_small(self):
        model = RegionKMeansHeuristic(
            self.data_small, self.reg_small, self.w_small, seed=RANDOM_STATE
        )
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.known_labels_small)

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_region_k_means_heuristic_synth_large(self):
        model = RegionKMeansHeuristic(
            self.data_large, self.reg_large, self.w_large, seed=RANDOM_STATE
        )
        model.solve()

        labs_ = model.labels_[: self.limit_index]
        numpy.testing.assert_equal(labs_, self.known_labels_large)

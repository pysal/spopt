import geopandas
import libpysal
import numpy
import pytest
from shapely.geometry import Polygon, box
from spopt.region import MaxPHeuristic
from spopt.region.maxp import infeasible_components, modify_components, plot_components
from spopt.region.base import form_single_component


# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestMaxPHeuristic:
    def setup_method(self):

        self.mexico = MEXICO.copy()
        self.mexico["count"] = 1
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)
        # labels for non-verbose and verbose:
        # count=1, threshold=4, top_n=2
        self.basic_labels = [5, 5, 3, 6, 6, 6, 1, 1, 1, 1, 8, 6, 8, 2, 2, 8]
        self.basic_labels += [2, 8, 7, 7, 2, 7, 5, 3, 3, 5, 3, 4, 4, 4, 4, 7]

        # labels for:
        # count=4, threshold=10, top_n=5
        self.complex_labels = [2, 2, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 3, 3]
        self.complex_labels += [1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2]

        # labels for one variable column:
        # count=1, threshold=5, top_n=5
        self.var1_labels = [3, 3, 6, 2, 2, 2, 2, 4, 2, 4, 5, 2, 5, 1, 1, 5, 1]
        self.var1_labels += [4, 5, 5, 1, 1, 3, 3, 6, 3, 6, 4, 4, 6, 6, 4]

        # components
        n_cols = 5
        n_rows = 10
        b = 0
        h = w = 10
        component_0 = [box(l * w, b, l * w + w, b + h) for l in range(n_cols)]
        b = b + h * 2
        component_1 = [
            box(l * w, b + h * r, l * w + w, b + h + h * r)
            for r in range(n_rows)
            for l in range(n_cols)
        ]
        geometries = component_0 + component_1

        self.gdf = geopandas.GeoDataFrame(
            geometry=geometries,
            data=numpy.ones((n_cols * n_rows + n_cols, 1), int),
            columns=["var"],
        )

    def test_maxp_heuristic_basic_nonverbose(self):
        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        threshold = 4
        top_n = 2
        threshold_name = "count"
        numpy.random.seed(123456)
        args = (self.mexico, self.w, attrs_name, threshold_name, threshold, top_n)
        model = MaxPHeuristic(*args)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.basic_labels)

    def test_maxp_heuristic_basic_verbose(self):
        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        threshold = 4
        top_n = 2
        threshold_name = "count"
        numpy.random.seed(123456)
        args = (self.mexico, self.w, attrs_name, threshold_name, threshold, top_n)
        model = MaxPHeuristic(*args, verbose=True)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.basic_labels)

    def test_maxp_heuristic_complex(self):
        attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        threshold = 10
        top_n = 5
        threshold_name = "count"
        numpy.random.seed(1999)
        args = (self.mexico, self.w, attrs_name, threshold_name, threshold, top_n)
        model = MaxPHeuristic(*args)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.complex_labels)

    def test_maxp_one_var(self):
        attrs_name = ["PCGDP2000"]
        threshold = 5
        top_n = 5
        threshold_name = "count"
        numpy.random.seed(1999)
        args = (self.mexico, self.w, attrs_name, threshold_name, threshold, top_n)
        model = MaxPHeuristic(*args)
        model.solve()

        numpy.testing.assert_array_equal(model.labels_, self.var1_labels)

    def test_infeasible_components(self):
        ifcs = infeasible_components(self.mexico, self.w, "count", 35)
        numpy.testing.assert_array_equal(ifcs, [0])

    def test_plot_components(self):
        nt = type(plot_components(self.mexico, self.w)).__name__
        assert nt == "Map"

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_modify_components(self):
        w = libpysal.weights.Queen.from_dataframe(self.gdf)
        gdf1, w1 = modify_components(self.gdf, w, "var", 6, policy="drop")
        assert gdf1.shape[0] == 50

        gdf1, w1 = modify_components(self.gdf, w, "var", 6, policy="single")
        assert gdf1.shape[0] == 55
        assert w1.neighbors[0] != w.neighbors[0]
        assert w1.neighbors[1] == w.neighbors[1]

        gdf1, w1 = modify_components(self.gdf, w, "var", 6, policy="multiple")
        assert gdf1.shape[0] == 55
        assert w1.neighbors[0] != w.neighbors[0]
        assert w1.neighbors[1] != w.neighbors[1]

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_modify_components_policy_error(self):
        policy = "Triple"
        with pytest.raises(ValueError, match=f"Unknown `policy`: '{policy}'"):
            modify_components(
                self.gdf,
                libpysal.weights.Queen.from_dataframe(self.gdf),
                "var",
                6,
                policy=policy,
            )

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_modify_components_xfeasible_error(self):
        with pytest.raises(ValueError, match="No feasible components found in input."):
            modify_components(
                self.gdf,
                libpysal.weights.Queen.from_dataframe(self.gdf),
                "var",
                100,
            )

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_form_single_component_already_single(self):
        _gdf = self.gdf[10:20].copy()
        w = libpysal.weights.Queen.from_dataframe(_gdf)
        single_component = form_single_component(_gdf, w)

        numpy.testing.assert_array_equal(
            numpy.zeros(_gdf.shape[0]), single_component.component_labels
        )

    @pytest.mark.filterwarnings("ignore:The weights matrix is not fully")
    def test_form_single_component_error(self):
        linkage = "Triple"
        with pytest.raises(ValueError, match=f"Unknown `linkage`: '{linkage}'"):
            form_single_component(
                self.gdf,
                libpysal.weights.Queen.from_dataframe(self.gdf),
                linkage=linkage,
            )

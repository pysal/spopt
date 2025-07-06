import geopandas
import libpysal
import numpy
import pytest
import sklearn
from packaging.version import Version

from spopt.region import SA3, extract_clusters

RANDOM_STATE = 12345
SKLEARN_L_16 = Version(sklearn.__version__) < Version("1.6.0")


@pytest.mark.skipif(SKLEARN_L_16, reason="does not work with old sklearn")
class TestSA3:
    def setup_method(self):
        pth = libpysal.examples.get_path("airbnb_Chicago 2015.shp")
        self.chicago = geopandas.read_file(pth)
        self.attrs_name = ["num_spots"]
        self.w = libpysal.weights.Queen.from_dataframe(self.chicago, use_index=False)

        self.eom_labels_3 = numpy.array(
            [
                3,
                3,
                12,
                3,
                3,
                2,
                11,
                -1,
                -1,
                2,
                7,
                7,
                7,
                -1,
                -1,
                2,
                6,
                6,
                6,
                -1,
                6,
                2,
                0,
                -1,
                0,
                -1,
                -1,
                -1,
                1,
                -1,
                2,
                -1,
                1,
                -1,
                -1,
                7,
                0,
                1,
                -1,
                11,
                9,
                -1,
                9,
                9,
                9,
                2,
                -1,
                10,
                10,
                9,
                9,
                10,
                8,
                12,
                12,
                12,
                0,
                -1,
                12,
                8,
                4,
                8,
                11,
                -1,
                4,
                4,
                11,
                0,
                11,
                11,
                5,
                9,
                5,
                5,
                6,
                2,
                -1,
            ]
        )

        self.leaf_labels_10 = numpy.array(
            [
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                0,
                0,
                0,
                -1,
                -1,
                -1,
                0,
                0,
                0,
                -1,
                0,
                -1,
                -1,
                -1,
                -1,
                0,
                0,
                -1,
                -1,
                0,
                -1,
                0,
                -1,
                -1,
                -1,
                0,
                -1,
                -1,
                -1,
                1,
                2,
                2,
                2,
                2,
                2,
                -1,
                -1,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                2,
                -1,
                -1,
                0,
                -1,
                0,
            ]
        )

    def test_ward_eom_3(self):
        numpy.random.seed(RANDOM_STATE)
        model = SA3(
            gdf=self.chicago, w=self.w, attrs_name=self.attrs_name, min_cluster_size=3
        )
        model.solve()
        numpy.testing.assert_equal(model.labels_, self.eom_labels_3)

    def test_ward_leaf_10(self):
        numpy.random.seed(RANDOM_STATE)
        model = SA3(
            gdf=self.chicago,
            w=self.w,
            attrs_name=self.attrs_name,
            min_cluster_size=10,
            extraction="leaf",
        )
        model.solve()
        numpy.testing.assert_equal(model.labels_, self.leaf_labels_10)

        # assert different from eom clusters
        model2 = SA3(
            gdf=self.chicago,
            w=self.w,
            attrs_name=self.attrs_name,
            min_cluster_size=10,
            extraction="eom",
        )
        model2.solve()
        assert not numpy.array_equal(model.labels_, model2.labels_)

    def test_extract_clusters(self):
        numpy.random.seed(RANDOM_STATE)
        model = SA3(
            gdf=self.chicago,
            w=self.w,
            attrs_name=self.attrs_name,
            min_cluster_size=10,
            extraction="leaf",
        )
        model.solve()

        graph = libpysal.graph.Graph.from_W(self.w)
        linkage_matrix = model._get_tree(
            self.chicago[self.attrs_name].values,
            graph.transform("B").sparse,
            clustering_kwds={"linkage": "ward", "metric": "euclidean"},
        )
        explicit_results = extract_clusters(
            linkage_matrix, min_cluster_size=10, extraction="leaf"
        )
        numpy.testing.assert_equal(model.labels_, explicit_results)
        numpy.testing.assert_equal(self.leaf_labels_10, explicit_results)

    def test_invalid_extraction(self):
        with pytest.raises(ValueError, match="Unsupported extraction method"):
            model = SA3(
                gdf=self.chicago,
                w=self.w,
                attrs_name=self.attrs_name,
                min_cluster_size=10,
                extraction="unsupported",
            )
            model.solve()

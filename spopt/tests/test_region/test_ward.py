import geopandas
import libpysal
import numpy
from packaging.version import Version

from spopt.region import WardSpatial

# see gh:spopt#437
LIBPYSAL_GE_48 = Version(libpysal.__version__) >= Version("4.8.0")
w_kwargs = {"use_index": True} if LIBPYSAL_GE_48 else {}


# Empirical tests -- Mexican states
RANDOM_STATE = 12345
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestWard:
    def setup_method(self):
        self.mexico = MEXICO.copy()
        self.mexico["count"] = 1
        self.attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico, **w_kwargs)
        self.known_labels = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 4, 0]
        self.known_labels += [1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0]

    def test_ward_defaults(self):
        numpy.random.seed(RANDOM_STATE)
        model = WardSpatial(gdf=self.mexico, w=self.w, attrs_name=self.attrs_name)
        model.solve()

        numpy.testing.assert_equal(model.labels_, self.known_labels)

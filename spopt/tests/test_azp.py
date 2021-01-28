import numpy as np
import libpysal
import geopandas as gpd
from spopt.region import AZP

def test_azp():
    pth = libpysal.examples.get_path("mexicojoin.shp")
    mexico = gpd.read_file(pth)
    mexico["count"] = 1
    attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
    w = libpysal.weights.Queen.from_dataframe(mexico)
    np.random.seed(123456)
    model = AZP(mexico, w, attrs_name, n_clusters=5)
    model.solve()
    labels = np.array(
        [0, 0, 2, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 4, 4, 2, 4, 3, 3, 3, 4, 3, 0, 0, 2, 2, 2, 1, 2, 1, 1, 4])
    return np.array_equal(model.labels_, labels)

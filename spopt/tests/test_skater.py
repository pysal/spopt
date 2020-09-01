import numpy as np
import libpysal
import geopandas as gpd

#from .. import SKATER

def test_skater():
    pth = libpysal.examples.get_path("mexicojoin.shp")
    mexico = gpd.read_file(pth)
    mexico["count"] = 1
    attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
    w = libpysal.weights.Queen.from_dataframe(mexico)
    np.random.seed(123456)
    model = SKATER(gdf=mexico, w=w, attrs_name=attrs_name)
    model.solve()
    labels = np.array(
        [0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
    return np.array_equal(model.labels_, labels)
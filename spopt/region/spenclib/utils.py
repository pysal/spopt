import scipy.sparse.csgraph as csg
import scipy.sparse as sp
from warnings import warn as Warn
import numpy as np


def check_weights(W, X=None, transform=None):
    if X is not None:
        assert (
            W.shape[0] == X.shape[0]
        ), "W does not have the same number of samples as X"
    graph = sp.csc_matrix(W)
    graph.eliminate_zeros()
    components, labels = csg.connected_components(graph)
    if components > 1:
        Warn(
            "Spatial affinity matrix is disconnected, and has {} subcomponents."
            "This will certainly affect the solution output."
        )
    return W


def lattice(x, y):
    """
    Construct a lattice of unit squares of dimension (x,y)
    """
    from shapely.geometry import Polygon
    import geopandas as gpd

    x = np.arange(x) * 1.0
    y = np.arange(y) * 1.0
    pgons = []
    for i in x:
        for j in y:
            ll, lr, ur, ul = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            # print([ll,lr,ur,ul])
            pgons.append(Polygon([ll, lr, ur, ul]))
    return gpd.GeoDataFrame({"geometry": pgons})

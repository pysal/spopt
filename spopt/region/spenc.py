from ..BaseClass import BaseSpOptHeuristicSolver
from .spenclib import SPENC


class Spenc(BaseSpOptHeuristicSolver):
    """
    Spatially encouraged spectral clustering.
    :cite:`wolf2018`
    """

    def __init__(self, gdf, w, attrs_name, n_clusters=5, random_state=None, gamma=1):
        """

        Parameters
        ----------

        gdf : geopandas.GeoDataFrame
            Input data.
            
        w : libpywal.weights.W
            Spatial weights matrix.

        attrs_name : list
            Strings for attribute names
            (columns of ``geopandas.GeoDataFrame``).

        n_clusters : int
            The number of clusters to form. Default is ``5``.
        
        random_state : int
            Random seed to set for reproducible results.
            Default is ``None``.
            
        gamma: int
            Default is ``1``.

        """
        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def solve(self):
        """Solve the spenc"""
        data = self.gdf
        X = data[self.attrs_name].values
        model = SPENC(
            n_clusters=self.n_clusters, random_state=self.random_state, gamma=self.gamma
        )
        model.fit(X, self.w.sparse)
        self.labels_ = model.labels_

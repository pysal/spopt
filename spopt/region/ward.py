from ..BaseClass import BaseSpOptHeuristicSolver
from warnings import warn
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)

class WardSpatial(BaseSpOptHeuristicSolver):
    """ Agglomerative clustering using Ward linkage with a spatial connectivity constraint.
        
        Parameters
        ----------
        
        gdf : geopandas.GeoDataFrame, required
            Geodataframe containing original data

        w : libpysal.weights.W, required
            Weights object created from given data

        attrs_name : list, required
            Strings for attribute names (cols of ``geopandas.GeoDataFrame``).

        n_clusters : int, optional, default: 5
            The number of clusters to form.

        clustering_kwds: dictionary, optional, default: dict()
            Other parameters about clustering could be used in sklearn.cluster.AgglometariveClustering.
        
        Returns
        -------

        labels_ : numpy.array
            Cluster labels for observations.


        """


    def __init__(self, gdf, w, attrs_name, n_clusters=5, clustering_kwds=dict()):
        
        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.n_clusters = n_clusters
        self.clustering_kwds = clustering_kwds


    def solve(self):
        """Solve the Ward"""
        data = self.gdf
        X = data[self.attrs_name].values
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters, connectivity=self.w.sparse, linkage="ward", **self.clustering_kwds
        )
        model.fit(X)
        self.labels_ = model.labels_

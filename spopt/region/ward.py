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
    """ Agglomerative clustering using Ward linkage with a spatial connectivity constraint."""
    
    def __init__(self, gdf, w, attrs_name, n_clusters=5):
        """
        
        Parameters
        ----------
        
        gdf : geopandas.GeoDataFrame

        w : libpywal.weights.W instance
        spatial weights matrix

        n_clusters : int, optional, default: 5
        The number of clusters to form.

        """
        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.n_clusters = n_clusters


    def solve(self):
        """Solve the Ward"""
        data = self.gdf
        X = data[self.attrs_name].values
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters, connectivity=self.w.sparse, linkage="ward"
        )
        model.fit(X)
        self.labels_ = model.labels_

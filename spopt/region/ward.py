from ..BaseClass import BaseSpOptHeuristicSolver
from sklearn.cluster import AgglomerativeClustering


class WardSpatial(BaseSpOptHeuristicSolver):
    """Agglomerative clustering using Ward linkage with a spatial connectivity constraint.

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


    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spopt.region import WardSpatial

    Read the data.

    >>> pth = libpysal.examples.get_path('airbnb_Chicago 2015.shp')
    >>> chicago = gpd.read_file(pth)

    Initialize the parameters.

    >>> w = libpysal.weights.Queen.from_dataframe(chicago)
    >>> attrs_name = ['num_spots']
    >>> n_clusters = 8

    Run the skater algorithm.

    >>> model = WardSpatial(chicago, w, attrs_name, n_clusters)
    >>> model.solve()

    Get the region IDs for unit areas.

    >>> model.labels_

    Show the clustering results.

    >>> chicago['ward_new'] = model.labels_
    >>> chicago.plot(column='ward_new', categorical=True, figsize=(12,8), edgecolor='w')

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
            n_clusters=self.n_clusters,
            connectivity=self.w.sparse,
            linkage="ward",
            **self.clustering_kwds
        )
        model.fit(X)
        self.labels_ = model.labels_

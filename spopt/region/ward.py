from sklearn.cluster import AgglomerativeClustering

from ..BaseClass import BaseSpOptHeuristicSolver


class WardSpatial(BaseSpOptHeuristicSolver):
    """Agglomerative clustering using Ward
    linkage with a spatial connectivity constraint.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        Geodataframe containing original data.
    w : libpysal.weights.W
        Weights object created from given data.
    attrs_name : list
        Strings for attribute names (cols of ``geopandas.GeoDataFrame``).
    n_clusters : int (default 5)
        The number of clusters to form.
    clustering_kwds: dict
        Other parameters about clustering could be used in
        ``sklearn.cluster.AgglometariveClustering.``

    Returns
    -------

    labels_ : numpy.array
        Cluster labels for observations.


    Examples
    --------

    >>> import geopandas
    >>> import libpysal
    >>> import numpy
    >>> from spopt.region import WardSpatial

    Read the data.

    >>> libpysal.examples.load_example('AirBnB')
    >>> pth = libpysal.examples.get_path('airbnb_Chicago 2015.shp')
    >>> chicago = gpd.read_file(pth)

    Initialize the parameters.

    >>> w = libpysal.weights.Queen.from_dataframe(chicago)
    >>> attrs_name = ['num_spots']
    >>> n_clusters = 8

    Run the ``WardSpatial`` algorithm.

    >>> model = WardSpatial(chicago, w, attrs_name, n_clusters)
    >>> model.solve()

    Get the counts of region IDs for unit areas.

    >>> numpy.array(numpy.unique(model.labels_, return_counts=True)).T
    array([[ 0, 62],
           [ 1,  6],
           [ 2,  3],
           [ 3,  1],
           [ 4,  2],
           [ 5,  1],
           [ 6,  1],
           [ 7,  1]])

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

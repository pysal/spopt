from ..BaseClass import BaseSpOptHeuristicSolver
from .spenclib import SPENC


class Spenc(BaseSpOptHeuristicSolver):
    """
    Spatially encouraged spectral clustering found in :cite:`wolf2018`.

    Apply clustering to a projection of the normalized laplacian, using
    spatial information to constrain the clustering. In practice Spectral
    Clustering is very useful when the structure of the individual clusters
    is highly non-convex or more generally when a measure of the center and
    spread of the cluster is not a suitable description of the complete cluster.
    For instance when clusters are nested circles on the 2D plan.

    Spatially-Encouraged Spectral Clustering (*SPENC*) is useful for when
    there may be highly non-convex clusters or clusters with irregular
    topology in a geographic context. If a binary weights matrix is provided
    during fit, this method can be used to find weighted normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using either
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced :math:`d(X, X)`::

        numpy.exp(-gamma * d(X,X) ** 2)

    or a :math:`k`-nearest neighbors connectivity matrix. Alternatively,
    using ``precomputed``, a user-provided affinity matrix can be used.
    Read more in the ``scikit-learn`` user guide on spectral clustering.

    """

    def __init__(
        self,
        gdf,
        w,
        attrs_name,
        n_clusters=5,
        random_state=None,
        gamma=1,
        eigen_solver=None,
        n_init=10,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol=1e-9,
        assign_labels="discretize",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=1,
    ):
        """

        Parameters
        ----------

        gdf : geopandas.GeoDataFrame
            Input data.
        w : libpywal.weights.W
            Spatial weights matrix.
        attrs_name : list
            Strings for attribute names from columns in ``gdf``.
        n_clusters : int (default 5)
            The number of clusters to form.
        random_state : int or numpy.random.RandomState (default None)
            A pseudo random number generator used for the initialization of the lobpcg
            eigen vectors decomposition when ``eigen_solver='amg'`` and by the
            :math:`k`-Means initialization.  If ``int``, ``random_state`` is the seed
            used by the random number generator; If ``numpy.random.RandomState``,
            ``random_state`` is the random number generator; If ``None``,
            the random number generator is the numpy.random.RandomState
            instance used by ``numpy.random``.
        gamma : int, float (default 1)
            Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
            Ignored for ``affinity='nearest_neighbors'``.
        eigen_solver : str (default None)
            The eigenvalue decomposition strategy to use. Valid values include
            ``{'arpack', 'lobpcg', 'amg'}``. AMG requires ``pyamg`` to be installed,
            which may be faster on very large, sparse problems, but may also lead to
            instabilities. *Note* – ``eigen_solver`` is ignored unless fitting using
            the ``breakme`` flag in the ``.fit()`` method (so do not use then).
        n_init : int (default 10)
            The number of times the :math:`k`-means algorithm will be run with
            different centroid seeds. The final results will be the best output of
            ``n_init`` consecutive runs in terms of inertia.
        affinity : str, array-like, callable (default 'rbf')
            If a ``str``, valid values include
            ``{'nearest_neighbors', 'precomputed', 'rbf'}`` or one of the kernels
            supported by ``sklearn.metrics.pairwise_kernels``. Only kernels that
            produce similarity scores (non-negative values that increase with
            similarity) should be used. *This property is not checked
            by the clustering algorithm*.
        n_neighbors : int (default 10)
            The number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. Ignored for ``affinity='rbf'``.
        eigen_tol : float (default 1e-7)
            Stopping criterion for eigen decomposition of the Laplacian matrix
            when using ``'arpack'`` as the ``eigen_solver``.
        assign_labels : str (default 'discretize')
            The strategy to use to assign labels in the embedding
            space. There are three ways to assign labels after the laplacian
            embedding: ``{'kmeans', 'discretize', 'hierarchical'}``:

            * ``'kmeans'`` can be applied and is a popular choice. But it can also be sensitive to initialization.
            * ``'discretize'`` is another approach which is less sensitive to random initialization, and which usually finds better clusters.
            * ``'hierarchical'`` decomposition repeatedly bi-partitions the graph, instead of finding the decomposition all at once, as suggested in :cite:`shi_malik_2000`.

        degree : float (default 3)
            Degree of the polynomial affinity kernel. Ignored by other kernels.
        coef0 : float (default 1)
            Zero coefficient for polynomial and sigmoid affinity kernels.
            Ignored by other kernels.
        kernel_params : dict (default None)
            Parameters (keyword arguments) and values for affinity kernel passed as
            callable object. Ignored by other affinity kernels.
        n_jobs : int (default 1)
            The number of parallel jobs to run for the nearest-neighbors
            affinity kernel, if used. If ``-1``, then the number of jobs
            is set to the number of CPU cores.

        Attributes
        ----------

        affinity_matrix_ : array-like
            Affinity matrix used for clustering in the shape of
            ``(n_samples, n_samples)``. Available only if after calling ``fit``.
        labels_ : list
            Cluster labels of each point or area.

        Notes
        -----

        If you have an affinity matrix, such as a distance matrix,
        for which ``0`` means identical elements, and high values mean
        very dissimilar elements, it can be transformed in a
        similarity matrix that is well suited for the algorithm by
        applying the Gaussian (RBF, heat) kernel::

            numpy.exp(-dist_matrix ** 2 / (2. * delta ** 2))

        Where ``delta`` is a free parameter representing the width of the Gaussian
        kernel.

        Another alternative is to take a symmetric version of the
        :math:`k`-nearest neighbors connectivity matrix of the points/areas.

        References
        ----------

        - :cite:`shi_malik_2000` Normalized cuts and image segmentation, 2000
          Jianbo Shi, Jitendra Malik – https://doi.org/10.1109/34.868688

        - :cite:`von2007tutorial` A Tutorial on Spectral Clustering, 2007
          Ulrike von Luxburg – https://doi.org/10.1007/s11222-007-9033-z

        - :cite:`yu_shi_2003` Multiclass spectral clustering, 2003
          Stella X. Yu, Jianbo Shi – https://doi.org/10.1109/ICCV.2003.1238361

        """  # noqa E402

        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_init = n_init
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def solve(self, fit_kwargs=dict()):
        """Solve the spenc.

        Parameters
        ----------

        fit_kwargs : dict
            Keyword arguments passed into ``spenclib.abstracts.SPENC.fit()``.

        """

        data = self.gdf
        X = data[self.attrs_name].values
        model = SPENC(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            gamma=self.gamma,
            eigen_solver=self.eigen_solver,
            n_init=self.n_init,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            n_jobs=self.n_jobs,
        )
        model.fit(X, self.w.sparse, **fit_kwargs)
        self.labels_ = model.labels_

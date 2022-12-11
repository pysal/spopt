from sklearn import cluster as clust
import sklearn.metrics as skm
import sklearn.metrics.pairwise as pw
from sklearn.utils.validation import check_array
from .utils import check_weights
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils import check_random_state
from sklearn.cluster._spectral import discretize as _discretize
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .scores import boundary_fraction
import scipy.sparse as spar
from scipy.sparse import csgraph as cg, linalg as la
from warnings import warn as Warn


class SPENC(clust.SpectralClustering):
    def __init__(
        self,
        n_clusters=8,
        eigen_solver=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
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
        Apply clustering to a projection of the normalized laplacian, using
        spatial information to constrain the clustering.

        In practice Spectral Clustering is very useful when the structure of
        the individual clusters is highly non-convex or more generally when
        a measure of the center and spread of the cluster is not a suitable
        description of the complete cluster. For instance when clusters are
        nested circles on the 2D plan.

        Spatially-Encouraged Spectral Clustering (SPENC) is useful for when
        there may be highly non-convex clusters or clusters with irregular
        topology in a geographic context.

        If a binary weights matrix is provided during fit, this method can be
        used to find weighted normalized graph cuts.

        When calling ``fit``, an affinity matrix is constructed using either
        kernel function such the Gaussian (aka RBF) kernel of the euclidean
        distanced ``d(X, X)``::

                numpy.exp(-gamma * d(X,X) ** 2)

        or a :math:`k`-nearest neighbors connectivity matrix.

        Alternatively, using ``precomputed``, a user-provided affinity
        matrix can be used. Read more in the ``scikit-learn`` user guide
        on spectral clustering.

        Parameters
        -----------

        n_clusters : int (default 5)
            The number of clusters to form.
        eigen_solver : str (default None)
            The eigenvalue decomposition strategy to use. Valid values include
            ``{'arpack', 'lobpcg', 'amg'}``. AMG requires ``pyamg`` to be installed,
            which may be faster on very large, sparse problems, but may also lead to
            instabilities. *Note* – ``eigen_solver`` is ignored unless fitting using
            the ``breakme`` flag in the ``.fit()`` method (so do not use then).
        random_state : int or numpy.random.RandomState (default None)
            A pseudo random number generator used for the initialization of the lobpcg
            eigen vectors decomposition when ``eigen_solver='amg'`` and by the
            :math:`k`-Means initialization.  If ``int``, ``random_state`` is the seed
            used by the random number generator; If ``numpy.random.RandomState``,
            ``random_state`` is the random number generator; If ``None``,
            the random number generator is the numpy.random.RandomState
            instance used by ``numpy.random``.
        n_init : int (default 10)
            The number of times the :math:`k`-means algorithm will be run with
            different centroid seeds. The final results will be the best output of
            ``n_init`` consecutive runs in terms of inertia.
        gamma : float, default=1.0
            Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
            Ignored for ``affinity='nearest_neighbors'``.
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

        - Normalized cuts and image segmentation, 2000
          Jianbo Shi, Jitendra Malik
          http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

        - A Tutorial on Spectral Clustering, 2007
          Ulrike von Luxburg
          http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

        - Multiclass spectral clustering, 2003
          Stella X. Yu, Jianbo Shi
          http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

        """

        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(
        self,
        X,
        W=None,
        y=None,
        shift_invert=True,
        breakme=False,
        check_W=True,
        grid_resolution=100,
        floor=0,
        floor_weights=None,
        cut_method="gridsearch",
    ):
        """Creates an affinity matrix for X using the selected affinity,
        applies W to the affinity elementwise, and then applies spectral clustering
        to the affinity matrix.

        Parameters
        ----------

        X               : sparse or dense array
                          matrix containing P features for N observations.
        W               : sparse or dense array, default None
                          matrix expressing the pairwise spatial relationships
                          between N observations.
        y               : sparse or dense array, default None
                          ignored, for scikit-learn class inheritance/regularity purposes.
        shift_invert    : bool, default True
                          boolean governing whether or not to use shift-invert
                          trick to finding sparse eigenvectors
        breakme         : bool, default False
                          Whether or not to simply pipe down to the sklearn spectral
                          clustering class. Will likely break the formal guarantees
                          about contiguity/connectedness of solutions, due to the
                          standardizations/short cuts taken in sklearn.cluster.SpectralClustering
        check_W         : bool, default True
                          Whether or not to check that the spatial weights matrix
                          is correctly formatted and aligns with the X matrix.
        grid_resolution : int, default 100
                          how many subdivisions to use when doing gridsearch
                          for cutpoint on second eigenvector of subgraphs.
        floor           : float/int, default 0
                          value which governs the lower limit on the size of partitions
                          if 0, there is no limit.
                          if floor_weights are provided, floor should be a limit on
                          the sum of floor weights for each region.
        floor_weights   : np.ndarray of shape (n,), default np.ones((n,))
                          array containing weights for each observation used to determine
                          the region floor.
        cut_method      : str, default 'gridsearch'
                          option governing what method to use to partition regions
                          1. "gridsearch" (default): the hierarchical grid search
                            suggested by Shi & Malik (2000); search the second
                            eigenvector for the "best" partition in terms of cut weight.
                          2. "zero": cut the eigenvector at zero. Usually a passable solution,
                            since the second eigenvector is usually centered around zero.
                          3. "median": cut the eigenvector through its median. This means the
                            regions will always be divided into two halves with equal numbers
                            of elemental units.
                          "gridsearch" may be slow when grid_resolution is large.
                          "zero" is the best method for large data.

        Notes
        -----

        breakme sends the affinity matrix down to scikit's spectral clustering class.
        I call this breakme because of bug8129.
        I don't see a significant difference here when switching between the two,
        most assignments in the problems I've examined are the same.
        I think, since the bug is in the scaling of the eigenvectors, it's not super important.

        But, in the future, it may make sense to investigate whether the bug in sklearn
        is fully fixed, which would mean that any spectral clustering for
        a weights matrix in sklearn would always be contiguous.

        """
        if np.isinf(self.n_clusters):
            self.assign_labels = "hierarchical"

        if X is not None:
            X = check_array(
                X,
                accept_sparse=["csr", "coo", "csc"],
                dtype=np.float64,
                ensure_min_samples=2,
            )
            if check_W:
                W = check_weights(W, X)

            if self.affinity == "nearest_neighbors":
                connectivity = kneighbors_graph(
                    X,
                    n_neighbors=self.n_neighbors,
                    include_self=True,
                    n_jobs=self.n_jobs,
                )
                self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            elif self.affinity == "precomputed":
                self.affinity_matrix_ = X
            else:
                params = self.kernel_params
                if params is None:
                    params = {}
                if not callable(self.affinity):
                    params["gamma"] = self.gamma
                    params["degree"] = self.degree
                    params["coef0"] = self.coef0
                self.attribute_affinity_ = pw.pairwise_kernels(
                    X, metric=self.affinity, filter_params=True, **params
                )
                self.spatial_affinity_ = W
                self.affinity_matrix_ = W.multiply(self.attribute_affinity_)
        else:
            self.affinity_matrix_ = W
        if breakme:  ##sklearn/issues/8129
            self.affinity_ = self.affinity
            self.affinity = "precomputed"
            super().fit(self.affinity_matrix_)

            self.affinity = self.affinity_
            del self.affinity_
            return self

        if self.assign_labels == "hierarchical":
            self.labels_ = self._spectral_bipartition(
                grid_resolution=grid_resolution,
                shift_invert=shift_invert,
                floor=floor,
                floor_weights=floor_weights,
            )
            return self

        embedding = self._embed(self.affinity_matrix_, shift_invert=shift_invert)
        self.embedding_ = embedding.T
        random_state = check_random_state(self.random_state)

        if self.assign_labels == "kmeans":
            self.labels_ = (
                clust.KMeans(n_clusters=self.n_clusters).fit(self.embedding_).labels_
            )
        else:
            self.labels_ = _discretize(self.embedding_, random_state=random_state)
        return self

    def _embed(self, affinity, shift_invert=True):
        """
        Compute the eigenspace embedding of a given affinity matrix.

        Parameters
        ----------

        affinity    :   sparse or dense matrix
                        affinity matrix to compute the spectral embedding of
        shift_invert:   bool
                        whether or not to use the shift-invert eigenvector search
                        trick useful for finding sparse eigenvectors.
        """
        laplacian, orig_d = cg.laplacian(affinity, normed=True, return_diag=True)
        laplacian *= -1
        random_state = check_random_state(self.random_state)
        v0 = random_state.uniform(-1, 1, laplacian.shape[0])

        if not shift_invert:
            ev, spectrum = la.eigsh(
                laplacian, which="LA", k=self.n_clusters, v0=v0, tol=self.eigen_tol
            )
        else:
            ev, spectrum = la.eigsh(
                laplacian,
                which="LM",
                sigma=1,
                k=self.n_clusters,
                v0=v0,
                tol=self.eigen_tol,
            )

        embedding = spectrum.T[self.n_clusters :: -1]  # sklearn/issues/8129
        embedding = embedding / orig_d
        embedding = _deterministic_vector_sign_flip(embedding)
        return embedding

    def _spectral_bipartition(
        self,
        grid_resolution=100,
        shift_invert=True,
        floor=0,
        floor_weights=None,
        cut_method="gridsearch",
    ):
        """
        Implements the recursive spectral bipartitioning of shi and malik (2000)
        If n_clusters = np.inf and floor > 0, then will find
        all possible cuts with more than X units.

        Parameters
        ----------

        grid_resolution : int
                          how many subdivisions to use when doing gridsearch
                          for cutpoint on second eigenvector of subgraphs.
                          (Default: 100)
        shift_invert    : bool
                          boolean governing whether or not to use shift-invert
                          trick to finding sparse eigenvectors
                          (Default: True)
        floor           : float/int
                          value which governs the lower limit on the size of partitions
                          if 0, there is no limit.
                          if floor_weights are provided, floor should be a limit on
                          the sum of floor weights for each region.
                          (Default: 0)
        floor_weights   : np.ndarray of shape (n,)
                          array containing weights for each observation used to determine
                          the region floor.
                          (Default: np.ones((n,)))
        cut_method      : str
                          option governing what method to use to partition regions
                          1. "gridsearch" (default): the hierarchical grid search
                            suggested by Shi & Malik (2000); search the second
                            eigenvector for the "best" partition in terms of cut weight.
                          2. "zero": cut the eigenvector at zero. Usually a passable solution,
                            since the second eigenvector is usually centered around zero.
                          3. "median": cut the eigenvector through its median. This means the
                            regions will always be divided into two halves with equal numbers
                            of elemental units.
                          "gridsearch" may be slow when grid_resolution is large.
                          "zero" is the best method for large data.
        """
        if floor_weights is None:
            floor_weights = np.ones((self.affinity_matrix_.shape[0],))
        if spar.issparse(self.affinity_matrix_):
            self.affinity_matrix_ = self.affinity_matrix_.tocsr()
        threshold = self.n_clusters
        self.n_clusters = 2
        discovered = 1
        this_cut = np.ones((self.affinity_matrix_.shape[0],)).astype(bool)
        cuts = []
        accepted_cuts = []
        while discovered < threshold:
            current_affinity = self.affinity_matrix_[this_cut, :][:, this_cut]
            embedding = self._embed(current_affinity, shift_invert=shift_invert)
            second_eigenvector = embedding[1]
            new_cut, score_of_cut = self._make_hierarchical_cut(
                second_eigenvector,
                current_affinity,
                grid_resolution,
                cut_method=cut_method,
                floor=floor,
            )
            left_cut = this_cut.copy()
            left_cut[left_cut] *= new_cut
            right_cut = this_cut.copy()
            right_cut[right_cut] *= ~new_cut
            assert (
                len(this_cut) == len(left_cut) == len(right_cut)
            ), "Indexing Error in cutting!"
            if ((left_cut * floor_weights).sum() > floor) & (
                (right_cut * floor_weights).sum() > floor
            ):
                if (tuple(left_cut) not in accepted_cuts) & (
                    tuple(right_cut) not in accepted_cuts
                ):
                    cuts.append(left_cut)
                    accepted_cuts.append(tuple(left_cut))
                    cuts.append(right_cut)
                    accepted_cuts.append(tuple(right_cut))
            discovered += 1
            try:
                this_cut = cuts.pop(0)
            except IndexError:
                break
        accepted_cuts = np.vstack(accepted_cuts)
        labels = np.ones((accepted_cuts[0].shape[0],)) * -1.0
        for i, k in enumerate(np.flipud(accepted_cuts)):
            unassigned = labels == -1
            should_assign = unassigned & k
            labels[should_assign] = i
        return LabelEncoder().fit_transform(labels)

    def _make_hierarchical_cut(
        self,
        second_eigenvector,
        affinity_matrix,
        grid_resolution,
        cut_method="median",
        floor=0,
    ):
        """Compute a single hierarchical cut using one of the methods described in
        Shi and Malik (2000).
        """

        def mkobjective(second_eigenvector):
            """This makes a closure around the objective function given an eigenvector"""

            def objective(cutpoint):
                cut = second_eigenvector <= cutpoint
                assocA = affinity_matrix[cut].sum(axis=1).sum()
                assocB = affinity_matrix[~cut].sum(axis=1).sum()
                cutAB = affinity_matrix[cut, :][:, ~cut].sum(axis=1).sum() * 2
                score = cutAB / assocA + cutAB / assocB
                if np.isnan(score):
                    score = np.inf
                return score

            return objective

        objective = mkobjective(second_eigenvector)

        if cut_method == "gridsearch":
            support = np.linspace(
                *np.percentile(second_eigenvector, q=(2, 98)), num=grid_resolution
            )

            objective_surface = [objective(cutpoint) for cutpoint in support]
            cutpoint = support[np.argmin(objective_surface)]
            cut = second_eigenvector <= cutpoint
            return cut, np.min(objective_surface)
        elif cut_method == "median":
            median = np.median(second_eigenvector)
            score = objective(median)
            return second_eigenvector < median, score
        else:
            score = objective(0)
            return second_eigenvector < 0, score

    def score(
        self,
        X,
        W,
        labels=None,
        delta=0.5,
        attribute_score=skm.calinski_harabasz_score,
        spatial_score=boundary_fraction,
        attribute_kw=dict(),
        spatial_kw=dict(),
    ):
        """
        Computes the score of the given label vector on data in X using convex
        combination weight in delta.

        Parameters
        ----------

        X               : numpy array (N,P)
                          array of data classified into `labels` to score.
        W               : sparse array or numpy array (N,N)
                          array representation of spatial relationships
        labels          : numpy array (N,)
                          vector of labels aligned with X and W
        delta           : float
                          weight to apply to the attribute score.
                          Spatial score is given weight 1 - delta,
                          and attributes weight delta.
                          Default: .5
        attribute_score : callable
                          function to use to evaluate attribute homogeneity
                          Must have signature attribute_score(X,labels,**params)
                          Default: sklearn.metrics.calinski_harabasz_score
                                   (within/between deviation ratio)
        spatial_score   : callable
                          function to use to evaluate spatial regularity/contiguity.
                          Must have signature spatial_score(X,labels,**params)
                          Default: boundary_ratio(W,X,labels,**spatial_kw)
        """
        if labels is None:
            if not hasattr(self, "labels_"):
                raise Exception("Object must be fit in order to avoid passing labels.")
            labels = self.labels_
        labels = np.asarray(labels).flatten()
        attribute_score = attribute_score(X, labels, **attribute_kw)
        spatial_score = spatial_score(W, labels, X=X, **spatial_kw)
        return delta * attribute_score + (1 - delta) * spatial_score

    def _sample_gen(self, W, n_samples=1, affinity="rbf", distribution=None, **fit_kw):
        """
        NOTE: this is the lazy generator version of sample
        Compute random clusters using random eigenvector decomposition.
        This uses random weights in spectral decomposition to generate approximately-evenly populated
        random subgraphs from W.

        Parameters
        ----------

        W                : np.ndarray or scipy.sparse matrix
                           matrix encoding the spatial relationships between observations in the frame.
                           Must be strictly binary & connected to result in connected graphs correct behavior.
                           Mathematical properties of randomregions are undefined if not.
        n_samples        : int, default 1
                           integer describing how many samples to construct
        affinity         : string or callable, default is 'rbf'
                           passed down to the underlying SPENC class when spectral spatial clusters are found.
        distribution     : callable default is numpy.random.normal(0,1, size=(N,1))
                           function when called with no arguments that draws the random weights used to
                           generate the random regions. Must align with W.
        spenc_parameters : keyword arguments
                           extra arguments passed down to the SPENC class for further customization.
        """
        if distribution is None:
            distribution = lambda: np.random.normal(0, 1, size=(W.shape[0], 1))
        else:
            assert callable(distribution), "distribution is not callable!"
        for _ in range(n_samples):
            randomweights = distribution()
            fitted = self.fit(randomweights, W, **fit_kw)
            yield fitted.labels_

    def sample(self, W, n_samples=1, distribution=None, **fit_kw):
        """
        Compute random clusters using random eigenvector decomposition.
        This uses random weights in spectral decomposition to generate approximately-evenly populated
        random subgraphs from W.

        Parameters
        ----------

        W             : np.ndarray or scipy.sparse matrix
                        matrix encoding the spatial relationships between observations in the frame.
                        Must be strictly binary & connected to result in connected graphs correct behavior.
                        Mathematical properties of randomregions are undefined if not.
        n_samples     : int, default 1
                        integer describing how many samples to construct
        affinity      : string or callable, default is 'rbf'
                        passed down to the underlying SPENC class when spectral spatial clusters are found.
        distribution  : callable default is numpy.random.normal(0,1, size=(N,1))
                        function when called with no arguments that draws the random weights used to
                        generate the random regions. Must align with W.
        fit_kw        : keyword arguments
                        extra arguments passed down to the SPENC class for further customization.
        Returns
        -------

        labels corresponding to the input W that are generated at random.

        """
        result = np.vstack(
            [
                labels
                for labels in self._sample_gen(
                    W, n_samples=n_samples, distribution=distribution, **fit_kw
                )
            ]
        )
        if n_samples == 1:
            result = result.flatten()
        return result


class AgglomerativeClustering(clust.AgglomerativeClustering):
    def _sample_gen(self, n_samples=25, distribution=None):
        """
        sample random clusters with agglomerative clustering using random weights.
        """
        if distribution is None:
            distribution = lambda: np.random.normal(
                0, 1, size=(self.connectivity.shape[0], 1)
            )
        else:
            assert callable(distribution), "distribution is not callable!"
        for _ in range(n_samples):
            randomweights = distribution()
            fitted = self.fit(randomweights)
            yield fitted.labels_

    def sample(self, n_samples=1, distribution=None):
        """
        Compute random clusters using randomly-weighted agglomerative clustering.
        This uses random weights in agglomerative clustering decomposition to generate
        random subgraphs from W.

        Parameters
        ----------

        W                : np.ndarray or scipy.sparse matrix
                           matrix encoding the spatial relationships between observations in the frame.
                           Must be strictly binary & connected to result in connected graphs correct behavior.
                           Mathematical properties of randomregions are undefined if not.
        n_samples        : int
                           integer describing how many samples to construct
        distribution     : callable (default: np.random.normal(0,1))
                           a function that, when called with no arguments, returns the weights
                           used as fake data to randomize the graph.

        Returns
        -------

        labels corresponding to the input W that are generated at random.

        """
        return np.vstack(
            [
                labels
                for labels in self._sample_gen(
                    n_samples=n_samples, distribution=distribution
                )
            ]
        )

from ..BaseClass import BaseSpOptHeuristicSolver

from sklearn.metrics import pairwise as skm
from scipy.sparse import csgraph as cg
from scipy.optimize import OptimizeWarning
from collections import namedtuple
import time
import numpy as np
import copy
import warnings

deletion = namedtuple("deletion", ("in_node", "out_node", "score"))


class SpanningForest(object):
    def __init__(
        self,
        dissimilarity=skm.manhattan_distances,
        affinity=None,
        reduction=np.sum,
        center=np.mean,
        verbose=False,
    ):
        """
        Initialize the SKATER algorithm.

        Parameters
        ----------

        dissimilarity : callable (default sklearn.metrics.pairwise.manhattan_distances)
            A callable distance metric.
        affinity : callable (default None)
            A callable affinity metric between 0 and 1, which is inverted to provide a
            dissimilarity metric. Either ``affinity`` or ``dissimilarity`` should be
            provided. If both arguments are provided ``dissimilarity`` is chosen.
        reduction : callable (default numpy.sum())
            The reduction applied over all clusters to provide the map score.
        center : callable (default numpy.mean())
            The method for computing the center of each region in attribute space.
        verbose : bool, int (default False)
            Flag for how much output to provide to the user,
            in terms of print statements and progress bars. Set to ``1`` for
            minimal output and ``2`` for full output.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        reduction should yield some kind of score where larger values are
        *less desirable* than smaller values. Typically, this means we use addition.

        """

        if affinity is not None and dissimilarity is not None:
            warnings.warn(
                "Both the `affinity` and `dissimilarity` arguments "
                "were passed in. Defaulting `dissimilarity`.",
                UserWarning,
            )
            affinity = None

        if affinity is not None:
            # invert the 0,1 affinity to an unbounded positive dissimilarity
            metric = lambda x, y: -np.log(affinity(x, y))
        else:
            metric = lambda x, y: dissimilarity(x, y)

        self.metric = metric
        self.reduction = reduction
        self.center = center
        self.verbose = verbose

    def __repr__(self):
        return (
            f"Minimum_Spanning_Tree_Pruning(metric={self.metric}, "
            f"reduction={self.reduction}, center={self.center})"
        )

    def fit(
        self,
        n_clusters,
        W,
        data=None,
        quorum=-np.inf,
        trace=False,
        islands="increase",
    ):
        """

        Parameters
        ----------

        n_clusters : int
            The number of clusters to form.
        W : libpysal.weights.W
            A PySAL weights object created from given data expressing the neighbor
            relationships between observations. It must be symmetric and binary, for
            example: Queen/Rook, DistanceBand, or a symmetrized KNN.
        data : numpy.ndarray (default None)
            An array of shape :math:`(N,P)` with :math:`N`
            observations and :math:`P` features.
        quorum : int, float (default -numpy.inf)
            The floor on the size of regions.
        trace : bool (default False)
            Flag denoting whether to store intermediate
            labelings as the tree gets pruned.
        islands : str (default 'increase')
            Description of what to do with islands. If ``'ignore'``, the algorithm will
            discover ``n_clusters`` regions, treating islands as their own regions. If
            "increase", the algorithm will discover ``n_clusters`` regions,
            treating islands as separate from ``n_clusters``.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        problem *minimizes* the map dissimilarity. Therefore, lower scores are better.

        """
        if trace:
            self._trace = []
        if data is None:
            attribute_kernel = np.ones((W.n, W.n))
            data = np.ones((W.n, 1))
        else:
            attribute_kernel = self.metric(data, None)
        W.transform = "b"
        W = W.sparse
        start = time.time()

        super_verbose = self.verbose > 1
        start_W = time.time()
        dissim = W.multiply(attribute_kernel)
        dissim.eliminate_zeros()
        end_W = time.time() - start_W

        if super_verbose:
            print(f"Computing Affinity Kernel took {end_W:.2f}s")

        tree_time = time.time()
        MSF = cg.minimum_spanning_tree(dissim)
        tree_time = time.time() - tree_time
        if super_verbose:
            print(f"Computing initial MST took {tree_time:.2f}s")

        init_component_time = time.time()
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        init_component_time = time.time() - init_component_time

        if super_verbose:
            print(f"Computing connected components took {init_component_time:.2f}s.")

        if current_n_subtrees > 1:
            island_warnings = [
                (
                    f"Increasing `n_clusters` from {n_clusters} to "
                    f"{n_clusters + current_n_subtrees} in order to "
                    "account for islands."
                ),
                (
                    "Counting islands towards the remaining "
                    f"{n_clusters - current_n_subtrees} clusters."
                ),
            ]
            ignoring_islands = int(islands.lower() == "ignore")
            chosen_warning = island_warnings[ignoring_islands]
            warnings.warn(
                f"By default, the graph is disconnected! {chosen_warning}",
                OptimizeWarning,
                stacklevel=2,
            )
            if not ignoring_islands:
                n_clusters += current_n_subtrees
            _, island_populations = np.unique(current_labels, return_counts=True)
            if (island_populations < quorum).any():
                raise ValueError(
                    "Islands must be larger than the quorum. If not, drop the small "
                    "islands and solve for clusters in the remaining field."
                )
        if trace:
            self._trace.append((current_labels, deletion(np.nan, np.nan, np.inf)))
            if super_verbose:
                print(self._trace[-1])
        while current_n_subtrees < n_clusters:  # while we don't have enough regions
            best_deletion = self.find_cut(
                MSF,
                data,
                quorum=quorum,
                labels=None,
                target_label=None,
            )

            # if our search succeeds
            if np.isfinite(best_deletion.score):
                # accept the best move as *the* move
                if super_verbose:
                    print(f"making cut {best_deletion}...")
                MSF, current_n_subtrees, current_labels = self.make_cut(
                    *best_deletion, MSF=MSF
                )
            # otherwise, it means the MSF admits no further cuts (no backtracking here)
            else:
                current_n_subtrees, current_labels = cg.connected_components(
                    MSF, directed=False
                )
                warnings.warn(
                    (
                        "MSF contains no valid moves after finding "
                        f"{current_n_subtrees} subtrees. Decrease the "
                        f"size of your quorum to find the remaining "
                        f"{n_clusters - current_n_subtrees} subtrees."
                    ),
                    OptimizeWarning,
                    stacklevel=2,
                )
                self.current_labels_ = current_labels
                self.minimum_spanning_forest_ = MSF
                self._elapsed_time = time.time() - start
                return self
            if trace:
                self._trace.append((current_labels, best_deletion))

        self.current_labels_ = current_labels
        self.minimum_spanning_forest_ = MSF
        self._elapsed_time = time.time() - start
        return self

    def score(self, data, labels=None, quorum=-np.inf):
        """
        This yields a score for the data, given the labels provided.
        If no labels are provided, and the object has been fit, then
        the labels discovered from the previous fit are used.
        If ``quorum`` is not passed, it is assumed to be irrelevant.
        If ``quorum`` is passed and the labels do not meet ``quorum``,
        the score is ``inf``.

        Parameters
        ----------

        data : numpy.array
            An :math:`(N,P)` array of data on which to compute
            the score of the regions expressed in ``labels``.
        labels : numpy.array (default None)
            An :math:`(N,)` vector of labels expressing the
            classification of each observation into a region.
        quorum : int, float (default -numpy.inf)
            The floor on the size of regions, which can be -inf if there is
            no lower bound (default). Any region below quorum makes the score inf.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        problem *minimizes* the map dissimilarity. So, lower scores are better.

        """
        if labels is None:
            try:
                labels = self.current_labels_
            except AttributeError:
                raise ValueError(
                    "Labels not provided and ``MSF_Prune object`` "
                    "has not been fit to data yet."
                )

        assert data.shape[0] == len(labels), (
            f"Length of label array ({labels.shape[0]}) "
            f"does not match length of data ({data.shape[0]})!"
        )

        _, subtree_quorums = np.unique(labels, return_counts=True)
        n_subtrees = len(subtree_quorums)
        if (subtree_quorums < quorum).any():
            return np.inf
        part_scores = [
            self.reduction(
                self.metric(
                    data[labels == l],
                    self.center(data[labels == l], axis=0).reshape(1, -1),
                )
            )
            for l in range(n_subtrees)
        ]
        return self.reduction(part_scores).item()

    def find_cut(
        self,
        MSF,
        data=None,
        quorum=-np.inf,
        labels=None,
        target_label=None,
        make=False,
    ):
        """
        Find the best cut from the MSF.

        Parameters
        ----------

        MSF : scipy.sparse.csgraph.minimum_spanning_tree
            An :math:`(N,N)` scipy sparse matrix with zero elements removed.
            representing the adjacency matrix for the minimum spanning forest.
            It is constructed from ``sparse.csgraph.sparse_from_dense`` or using
            ``MSF.eliminate_zeros()``. You **MUST** remove zero entries for this
            to work, otherwise they are considered no-cost paths.
        data : numpy.array (default None)
            An :math:`(N,P)` attribute matrix. If not provided, it is
            replaced with an :math:`(N,1)` vector of ones.
        quorum : int, float (default -numpy.inf)
            The minimum number of elements in the region
        labels : numpy.array (default None)
            An :math:`(N,)` vector of labels expressing the classification of each
            observation into a region. This represents the "cluster labels"
            for disconnected components of the graph.
        target_label : int (default None)
            The target label from the labels array to subset the MSF. If passed along
            with ``labels``, then a cut will be found that is restricted
            to that subset of the ``MSF``.
        make : bool (default False)
            Whether or not to modify the input ``MSF`` in order
            to make the best cut that was found.

        Returns
        -------

        namedtuple
            A ``namedtuple`` with ``in_node``, ``out_node``, and ``score``.

        """
        if data is None:
            data = np.ones(MSF.shape)

        if (labels is None) != (target_label is None):
            raise ValueError(
                "Both ``labels`` and ``target_label`` must be supplied! Only "
                f"{['labels', 'target_label'][int(target_label is None)]} provided."
            )
        if self.verbose:
            try:
                from tqdm.auto import tqdm
            except ImportError:

                def tqdm(noop, desc=""):
                    return noop

        else:

            def tqdm(noop, desc=""):
                return noop

        zero_in = (labels is not None) and (target_label is not None)
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        best_deletion = deletion(np.nan, np.nan, np.inf)
        for in_node, out_node in tqdm(
            np.vstack(MSF.nonzero()).T, desc="finding cut..."
        ):  # iterate over MSF edges
            if zero_in:
                if labels[in_node] != target_label:
                    continue

            local_MSF = copy.deepcopy(MSF)

            # delete a candidate edge
            local_MSF[in_node, out_node] = 0
            local_MSF.eliminate_zeros()

            # get the connected components
            local_n_subtrees, local_labels = cg.connected_components(
                local_MSF, directed=False
            )

            if local_n_subtrees <= current_n_subtrees:
                raise ValueError(
                    "Malformed MSF! `local_n_subtrees <= current_n_subtrees`"
                )

            # compute the score of these components
            score = self.score(data, labels=local_labels, quorum=quorum)

            # if the score is lower than the best score and quorum is met
            if score < best_deletion.score:
                best_deletion = deletion(in_node, out_node, score)
        if make:
            return self.make_cut(*best_deletion, MSF=MSF)
        return best_deletion

    def make_cut(self, in_node, out_node, score, MSF=None):
        """
        Make a cut on the MSF inplace.

        Parameters
        ----------

        in_node : int
            The ID of the source node for the edge to be cut.
        out_node : int
            The ID of the destination node for the edge to be cut.
        score : float
            The value of the score being cut. If the score is
            infinite, the cut is not made.
        MSF : scipy.sparse.csgraph.minimum_spanning_tree (default None)
            The spanning forest to use when making the cut. If not provided,
            uses the default tree in ``self.minimum_spanning_forest_``.

        """
        if MSF is None:
            MSF = self.minimum_spanning_forest_
        if np.isfinite(score):
            MSF[in_node, out_node] = 0
            MSF.eliminate_zeros()
            return (MSF, *cg.connected_components(MSF, directed=False))
        raise OptimizeWarning(
            "Score of the ({},{}) cut is inf, the quorum is likely not met!"
        )


class Skater(BaseSpOptHeuristicSolver):
    """Skater is a spatial regionalization algorithm based on spanning tree pruning
    introduced in :cite:`assunccao2006efficient`.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        A Geodataframe containing original data. The ``data`` attribute is
        derived from ``gdf`` as the ``attrs_name`` columns.
    w : libpysal.weights.W
        A PySAL weights object created from given data expressing the neighbor
        relationships between observations. It must be symmetric and binary, for
        example: Queen/Rook, DistanceBand, or a symmetrized KNN.
    attrs_name : list
        Strings for attribute names (columns of ``geopandas.GeoDataFrame``).
    n_clusters : int (default 5)
        The number of clusters to form.
    floor : int, float (default -numpy.inf)
        The floor on the size of regions.
    trace : bool (default False)
        Flag denoting whether to store intermediate labelings as the tree gets pruned.
    islands : str (default 'increase')
        Description of what to do with islands. If ``'ignore'``, the algorithm will
        discover ``n_clusters`` regions, treating islands as their own regions. If
        "increase", the algorithm will discover ``n_clusters`` regions,
        treating islands as separate from ``n_clusters``.
    spanning_forest_kwds : dict (default dict())
        Keyword arguments to be passed to ``SpanningForest`` including
        ``dissimilarity``, ``affinity``, ``reduction``, and ``center``.
        See ``spopt.region.skater.SpanningForest`` for docstrings.

    Attributes
    ----------

    labels_ : numpy.array
        Region IDs for observations.

    Examples
    --------

    >>> from spopt.region import Skater
    >>> import geopandas
    >>> import libpysal
    >>> import numpy
    >>> from sklearn.metrics import pairwise as skm

    Read the data.

    >>> pth = libpysal.examples.get_path('airbnb_Chicago 2015.shp')
    >>> chicago = geopandas.read_file(pth)

    Initialize the parameters.

    >>> w = libpysal.weights.Queen.from_dataframe(chicago)
    >>> attrs_name = ['num_spots']
    >>> n_clusters = 10
    >>> floor = 3
    >>> trace = False
    >>> islands = 'increase'
    >>> spanning_forest_kwds = dict(
    ...     dissimilarity=skm.manhattan_distances,
    ...     affinity=None,
    ...     reduction=numpy.sum,
    ...     center=numpy.mean
    ... )

    Run the skater algorithm.

    >>> model = Skater(
    ...     chicago, w,
    ...     attrs_name,
    ...     n_clusters,
    ...     floor,
    ...     trace,
    ...     islands,
    ...     spanning_forest_kwds
    ... )
    >>> model.solve()

    Get the region IDs for unit areas.

    >>> model.labels_

    Show the clustering results.

    >>> chicago['skater_new'] = model.labels_
    >>> chicago.plot(column='skater_new', categorical=True, figsize=(12,8), edgecolor='w')

    """

    def __init__(
        self,
        gdf,
        w,
        attrs_name,
        n_clusters=5,
        floor=-np.inf,
        trace=False,
        islands="increase",
        spanning_forest_kwds=dict(),
    ):
        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.n_clusters = n_clusters
        self.floor = floor
        self.trace = trace
        self.islands = islands
        self.spanning_forest_kwds = spanning_forest_kwds

    def solve(self):
        data = self.gdf
        X = data[self.attrs_name].values
        model = SpanningForest(**self.spanning_forest_kwds)
        model.fit(
            self.n_clusters,
            self.w,
            data=X,
            quorum=self.floor,
            trace=self.trace,
            islands=self.islands,
        )
        self.labels_ = model.current_labels_

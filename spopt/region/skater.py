from ..BaseClass import BaseSpOptHeuristicSolver

from sklearn.metrics import pairwise as skm
from scipy.sparse import csgraph as cg
from scipy.optimize import OptimizeWarning
from collections import namedtuple
from warnings import warn
import time
import numpy as np
import copy

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

        dissimilarity : a callable distance metric
        affinity : an callable affinity metric between 0,1.
                   Will be inverted to provide a
                   dissimilarity metric.
        reduction: the reduction applied over all clusters
                   to provide the map score.
        center:    way to compute the center of each region in attribute space

        verbose: bool/int describing how much output to provide to the user,
                 in terms of print statements or progressbars.

        NOTE: Optimization occurs with respect to a *dissimilarity* metric, so the reduction should
              yield some kind of score where larger values are *less desirable* than smaller values.
              Typically, this means we use addition.
        """
        if affinity is not None:
            # invert the 0,1 affinity to
            # to an unbounded positive dissimilarity
            metric = lambda x: -np.log(affinity(x))
        else:
            metric = dissimilarity
        self.metric = metric
        self.reduction = reduction
        self.center = center
        self.verbose = verbose

    def __repr__(self):
        return "Minimum_Spanning_Tree_Pruning(metric = {}, reduction = {}, center = {})".format(
            self.metric, self.reduction, self.center
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
        n_clusters : int of clusters wanted
        W : pysal W object expressing the neighbor relationships between observations.
            Should be symmetric and binary, so Queen/Rook, DistanceBand, or a symmetrized KNN.
        data: np.ndarray of (N,P) shape with N observations and P features
        quorum: floor on the size of regions.
        trace: bool denoting whether to store intermediate
               labelings as the tree gets pruned
        islands: string describing what to do with islands.
                 If "ignore", will discover `n_clusters` regions, treating islands as their own regions.
                 If "increase", will discover `n_clusters` regions, treating islands as separate from n_clusters.

        NOTE: Optimization occurs with respect to a *dissimilarity* metric, so the problem *minimizes*
              the map dissimilarity. So, lower scores are better.
        """
        if trace:
            self._trace = []
        if data is None:
            attribute_kernel = np.ones((W.n, W.n))
            data = np.ones((W.n, 1))
        else:
            attribute_kernel = self.metric(data)
        W.transform = "b"
        W = W.sparse
        start = time.time()

        super_verbose = self.verbose > 1
        start_W = time.time()
        dissim = W.multiply(attribute_kernel)
        dissim.eliminate_zeros()
        end_W = time.time() - start_W

        if super_verbose:
            print("Computing Affinity Kernel took {:.2f}s".format(end_W))

        tree_time = time.time()
        MSF = cg.minimum_spanning_tree(dissim)
        tree_time = time.time() - tree_time
        if super_verbose:
            print("Computing initial MST took {:.2f}s".format(tree_time))

        initial_component_time = time.time()
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        initial_component_time = time.time() - initial_component_time

        if super_verbose:
            print(
                "Computing connected components took {:.2f}s.".format(
                    initial_component_time
                )
            )

        if current_n_subtrees > 1:
            island_warnings = [
                "Increasing `n_clusters` from {} to {} in order to account for islands.".format(
                    n_clusters, n_clusters + current_n_subtrees
                ),
                "Counting islands towards the remaining {} clusters.".format(
                    n_clusters - (current_n_subtrees)
                ),
            ]
            ignoring_islands = int(islands.lower() == "ignore")
            chosen_warning = island_warnings[ignoring_islands]
            warn(
                "By default, the graph is disconnected! {}".format(chosen_warning),
                OptimizeWarning,
                stacklevel=2,
            )
            if not ignoring_islands:
                n_clusters += current_n_subtrees
            _, island_populations = np.unique(current_labels, return_counts=True)
            if (island_populations < quorum).any():
                raise ValueError(
                    "Islands must be larger than the quorum. If not, drop the small islands and solve for"
                    " clusters in the remaining field."
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

            if np.isfinite(best_deletion.score):  # if our search succeeds
                # accept the best move as *the* move
                if super_verbose:
                    print("making cut {}...".format(best_deletion))
                MSF, current_n_subtrees, current_labels = self.make_cut(
                    *best_deletion, MSF=MSF
                )
            else:  # otherwise, it means the MSF admits no further cuts (no backtracking here)
                current_n_subtrees, current_labels = cg.connected_components(
                    MSF, directed=False
                )
                warn(
                    "MSF contains no valid moves after finding {} subtrees."
                    "Decrease the size of your quorum to find the remaining {} subtrees.".format(
                        current_n_subtrees, n_clusters - current_n_subtrees
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
        This yields a score for the data, given the labels provided. If no labels are provided,
        and the object has been fit, then the labels discovered from the previous fit are used.

        If a quorum is not passed, it is assumed to be irrelevant.

        If a quorum is passed and the labels do not meet quorum, the score is inf.

        data    :   (N,P) array of data on which to compute the score of the regions expressed in labels
        labels  :   (N,) array of labels expressing the classification of each observation into a region.
        quorum  :   int expressing the minimum size of regions. Can be -inf if there is no lower bound.
                    Any region below quorum makes the score inf.

        NOTE: Optimization occurs with respect to a *dissimilarity* metric, so the problem *minimizes*
              the map dissimilarity. So, lower scores are better.
        """
        if labels is None:
            try:
                labels = self.current_labels_
            except AttributeError:
                raise ValueError(
                    "Labels not provided and MSF_Prune object has not been fit to data yet."
                )
        assert data.shape[0] == len(
            labels
        ), "Length of label array ({}) does not match " "length of data ({})! ".format(
            labels.shape[0], data.shape[0]
        )
        _, subtree_quorums = np.unique(labels, return_counts=True)
        n_subtrees = len(subtree_quorums)
        if (subtree_quorums < quorum).any():
            return np.inf
        part_scores = [
            self.reduction(
                self.metric(
                    X=data[labels == l],
                    Y=self.center(data[labels == l], axis=0).reshape(1, -1),
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

        MSF: (N,N) scipy sparse matrix with zero elements removed.
             Represents the adjacency matrix for the minimum spanning forest.
             Constructed from sparse.csgraph.sparse_from_dense or using MSF.eliminate_zeros().
             You MUST remove zero entries for this to work, otherwise they are considered no-cost paths.
        data: (N,p) attribute matrix. If not provided, replaced with (N,1) vector of ones.
        quorum: int denoting the minimum number of elements in the region
        labels: (N,) flat vector of labels for each point. Represents the "cluster labels"
                for disconnected components of the graph.
        target_label: int from the labels array to subset the MSF. If passed along with `labels`, then a cut
                      will be found that is restricted to that subset of the MSF.
        make: bool, whether or not to modify the input MSF in order to make the best cut that was found.

        Returns a namedtuple with in_node, out_node, and score.
        """
        if data is None:
            data = np.ones(MSF.shape)

        if (labels is None) != (target_label is None):
            raise ValueError(
                "Both labels and target_label must be supplied! Only {} provided.".format(
                    ["labels", "target_label"][int(target_label is None)]
                )
            )
        if self.verbose:
            try:
                from tqdm import tqdm
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
                raise Exception("Malformed MSF!")

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
        make a cut on the MSF inplace, provided the in_node, out_node, MSF, and score.
        in_node: int, ID of the source node for the edge to be cut
        out_node: int, ID of the destination node for the edge to be cut
        score: float, the value of the score being cut. if the score is infinite, the cut is not made.
        MSF: the spanning forest to use when making the cut. If not provided,
             uses the defualt tree in self.minimum_spanning_forest_
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


    Parameters
    ----------

    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data.

    w : libpysal.weights.W, required
        Weights object created from given data.

    attrs_name : list, required
        Strings for attribute names (cols of ``geopandas.GeoDataFrame``).

    n_clusters : int, optional, default: 5
        The number of clusters to form.

    floor: floor on the size of regions, default: -inf

    trace: bool denoting whether to store intermediate, default: False
           labelings as the tree gets pruned.

    islands: string describing what to do with islands, default: "increase"
             If "ignore", will discover `n_clusters` regions, treating islands as their own regions.
             If "increase", will discover `n_clusters` regions, treating islands as separate from n_clusters.


    spanning_forest_kwds include:

    dissimilarity :
        A callable distance metric.

    affinity : an callable affinity metric between 0,1
        Will be inverted to provide a dissimilarity metric.

    reduction:
        The reduction applied over all clusters to provide the map score.

    center:
        A way to compute the center of each region in attribute space.

    Attributes
    -------

    labels_ : numpy.array
        Region IDs for observations.


    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spopt.region.skater import Skater
    >>> from sklearn.metrics import pairwise as skm

    Read the data.

    >>> pth = libpysal.examples.get_path('airbnb_Chicago 2015.shp')
    >>> chicago = gpd.read_file(pth)

    Initialize the parameters.

    >>> w = libpysal.weights.Queen.from_dataframe(chicago)
    >>> attrs_name = ['num_spots']
    >>> n_clusters = 10
    >>> floor = 3
    >>> trace = False
    >>> islands = "increase"
    >>> spanning_forest_kwds = dict(dissimilarity=skm.manhattan_distances, affinity=None, reduction=np.sum, center=np.mean)

    Run the skater algorithm.

    >>> model = Skater(chicago, w, attrs_name, n_clusters, floor, trace, islands, spanning_forest_kwds)
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

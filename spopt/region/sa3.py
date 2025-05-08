import numpy as np
import pandas as pd
from libpysal import graph, weights
from sklearn.cluster import AgglomerativeClustering

from spopt.BaseClass import BaseSpOptHeuristicSolver


def extract_clusters(linkage_matrix, min_cluster_size, extraction="eom"):
    """Extract cluster types from a linkage matrix.

    Parameters
    ----------

    linkage_matrix : np.ndarray
        A hierarchical clustering encoded as an array in
        ``scipy.cluster.hierarchy.linkage`` format.
    min_cluster_size : int
        The minimum number of observations that forms a cluster.
    extraction: str (default "eom")
        The cluster extraction scheme. "eom" is Excess of Mass, "leaf" is Leaf
        extraction.

    Returns
    -------
    numpy.array
        Cluster labels for observations.

    Note
    ----
    This function requires ``fast_hdbscan`` and ``numba``.

    """

    n_samples = linkage_matrix.shape[0] + 1

    try:
        from fast_hdbscan.cluster_trees import (
            cluster_tree_from_condensed_tree,
            condense_tree,
            extract_eom_clusters,
            extract_leaves,
            get_cluster_label_vector,
        )
    except ImportError as e:
        raise ImportError(
            "The fast_hdbscan and numba libraries are required for this functionality."
        ) from e

    condensed_tree = condense_tree(linkage_matrix, min_cluster_size=min_cluster_size)

    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    if extraction == "eom":
        selected_clusters = extract_eom_clusters(
            condensed_tree, cluster_tree, allow_single_cluster=False
        )
    elif extraction == "leaf":
        selected_clusters = extract_leaves(condensed_tree, allow_single_cluster=False)
    else:
        raise ValueError("Unsupported extraction method. Use one of ('eom', 'leaf').")

    return get_cluster_label_vector(condensed_tree, selected_clusters, 0, n_samples)


class SA3(BaseSpOptHeuristicSolver):
    """Spatial Adaptive Agglomerative Aggregation (SA3) clustering.

    The algorithm carries out ``sklearn.cluster.AgglometariveClustering``
    per the specified parameters and extracts clusters from it, using density-clustering
    extraction algorithms - Excess of Mass or Leaf. This results in multiscale,
    contiguous clusters with noise.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        Geodataframe containing original data.
    w : libpysal.weights.W | libpysal.graph.Graph
        Weights or Graph object created from given data.
    attrs_name : list
        Strings for attribute names (cols of ``geopandas.GeoDataFrame``).
    min_cluster_size : int
        The minimum number of observations to form a cluster.
    extraction: str (default "eom")
        The cluster extraction scheme. "eom" is Excess of Mass, "leaf" is Leaf
        extraction.
    **kwargs
        Additional keyword arguments to be used in
        ``sklearn.cluster.AgglometariveClustering.``

    Attributes
    ----------

    labels_ : numpy.array
        Cluster labels for observations.

    """

    def __init__(
        self, gdf, w, attrs_name, min_cluster_size=15, extraction="eom", **kwargs
    ):
        self.gdf = gdf

        if isinstance(w, weights.W):
            w = graph.Graph.from_W(w)
        elif not isinstance(w, graph.Graph):
            raise ValueError(
                "Unkown graph type. Pass either libpysal.graph.Graph "
                "or libpysal.weights.W."
            )

        self.w = w
        self.attrs_name = attrs_name
        self.min_cluster_size = min_cluster_size
        self.extraction = extraction
        self.clustering_kwds = kwargs
        if "linkage" not in self.clustering_kwds:
            self.clustering_kwds["linkage"] = "ward"
        if "metric" not in self.clustering_kwds:
            self.clustering_kwds["metric"] = "euclidean"

    def solve(self):
        """Compute the labels."""

        # label input data, could work with empty tess as well
        labels = self.w.component_labels

        results = []
        for label in np.unique(labels):
            component_members = labels[labels == label].index.values

            # there are few component members, label all as noise
            if component_members.shape[0] <= self.min_cluster_size:
                results.append(
                    pd.Series(
                        np.full(component_members.shape[0], -1), index=component_members
                    )
                )
                continue

            component_graph = self.w.subgraph(component_members)
            component_data = self.gdf.loc[component_members, self.attrs_name]
            component_tree = self._get_tree(
                component_data,
                component_graph.transform("B").sparse,
                self.clustering_kwds,
            )

            # # sometimes ward/average linkage breaks the monotonic increase in the MST
            # # if that happens shift all distances by the max drop
            # # need a loop because several connections might be problematic
            problem_idxs = np.where(component_tree[1:, 2] < component_tree[0:-1, 2])[0]
            while problem_idxs.shape[0]:
                component_tree[problem_idxs + 1, 2] = (
                    component_tree[problem_idxs, 2] + 0.01
                )
                problem_idxs = np.where(
                    component_tree[1:, 2] < component_tree[0:-1, 2]
                )[0]
            # check if tree distances are always increasing
            assert (component_tree[1:, 2] >= component_tree[0:-1, 2]).all()

            component_clusters = extract_clusters(
                component_tree, self.min_cluster_size, extraction=self.extraction
            )

            results.append(pd.Series(component_clusters, index=component_members))

        # relabel local clusters [[0,1,2], [0,1]] to global clusters [0, 1, 2, 3, 4],
        # while keeping noise labels as they are
        new_labels = []
        next_cluster_val = 0
        for labels_set in results:
            is_noise = labels_set == -1
            # if this component is all noise, add it but skip the cluster increment
            if is_noise.all():
                new_labels.append(labels_set)
                continue

            highest_cluster_count = labels_set.max()
            labels_set[~is_noise] = labels_set[~is_noise] + next_cluster_val
            new_labels.append(labels_set)
            next_cluster_val += highest_cluster_count + 1

        # set the labels in the same order as the input data
        self.labels_ = pd.concat(new_labels).loc[self.gdf.index]

    def _get_tree(self, training_data, clustering_graph, clustering_kwds):
        """Carry the agglomerative clustering, transform the result
        and return the linkage matrix."""

        clusterer = AgglomerativeClustering(
            connectivity=clustering_graph,
            metric=clustering_kwds["metric"],
            linkage=clustering_kwds["linkage"],
            compute_full_tree=True,
            compute_distances=True,
        )
        model = clusterer.fit(training_data)

        # Create a linkage matrix from a sklearn hierarchical clustering model.
        # by getting the counts for every connection
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        return linkage_matrix

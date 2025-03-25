# ruff: noqa: B006, C408

from sklearn.cluster import AgglomerativeClustering

from spopt.BaseClass import BaseSpOptHeuristicSolver

from numpy import full, zeros, unique, where, column_stack
from pandas import Series, concat
from libpysal.graph import Graph
from libpysal.weights import W

def extract_clusters(linkage_matrix, min_cluster_size, eom_clusters=True):
    '''Extract hdbscan cluster types from a linkage matrix.'''
    
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
        raise "The fast_hdbscan and numba librarries are required for this functionality."

    condensed_tree = condense_tree(linkage_matrix, 
                            min_cluster_size=min_cluster_size)
    
    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    if eom_clusters:
        selected_clusters = extract_eom_clusters(
            condensed_tree, cluster_tree, allow_single_cluster=False
        )
    else:
        selected_clusters = extract_leaves(
                condensed_tree, allow_single_cluster=False
            )
        
    return get_cluster_label_vector(condensed_tree, selected_clusters, 0, n_samples)


class SA3(BaseSpOptHeuristicSolver):
    """Spatial Adaptive Agglomerative Aggregation. SA^3
    
    Note: This is a conservative clustering in general. Leaf is more conservative than EOM.
    """

    def __init__(self, gdf, w, attrs_name, min_cluster_size=15, eom_clusters=True, clustering_kwds=dict()):

        self.gdf = gdf

        if isinstance(w, W):
            w = Graph.from_W(w)
        elif not isinstance(w, W) and not isinstance(w, Graph):
            raise "Unkown graph type."
        
        self.w = w
        self.attrs_name = attrs_name
        self.min_cluster_size = min_cluster_size
        self.eom_clusters = eom_clusters
        self.clustering_kwds = clustering_kwds
        if 'linkage' not in self.clustering_kwds:
            self.clustering_kwds['linkage'] = 'ward'
        if 'metric' not in self.clustering_kwds:
            self.clustering_kwds['metric'] = 'euclidean'

    def solve(self):
        '''Split the input data into connected components and carry out an agglomerative clustering for each component independently, then combine all the seperate clusterings into one set.'''
        
        # label input data, could work with empty tess as well
        labels = self.w.component_labels
        
        ### have to be careful about assigning labels to the variables and be careful of noise.
        results = []
        
        for label in unique(labels):

            component_members = labels[labels == label].index.values

            # there are few component members, label all as noise
            if component_members.shape[0] <= self.min_cluster_size:
                results.append(
                    Series(full(component_members.shape[0], -1),
                           index=component_members))
                continue
        
            component_graph = self.w.subgraph(component_members)
            component_data = self.gdf.loc[component_members, self.attrs_name]
            component_tree = self._get_tree(component_data, 
                                            component_graph.transform('B').sparse,
                                            self.clustering_kwds)
    
            # # sometimes ward/average linkage breaks the monotonic increase in the MST
            # # if that happens shift all distances by the max drop
            # # need a loop because several connections might be problematic
            problem_idxs = where(component_tree[1:, 2] < component_tree[0:-1, 2])[0]
            while problem_idxs.shape[0]:
                component_tree[problem_idxs + 1, 2] = component_tree[problem_idxs, 2] + .01
                problem_idxs = where(component_tree[1:, 2] < component_tree[0:-1, 2])[0]
            # check if tree distances are always increasing
            assert (component_tree[1:, 2] >= component_tree[0:-1, 2]).all()
            
            component_clusters = extract_clusters(component_tree, 
                                              self.min_cluster_size, 
                                              eom_clusters=self.eom_clusters)
            
            results.append(Series(component_clusters, index=component_members))

        # relabel local clusters - [[0,1,2], [0,1]] to global clusters  - [0, 1, 2, 3, 4]
        # while keeping noise labels as they are
        new_labels = []
        next_cluster_val = 0
        for labels_set in results:
            not_noise = labels_set != - 1
            labels_set[not_noise] = labels_set[not_noise] + next_cluster_val
            new_labels.append(labels_set)
            
            next_cluster_val += (labels_set.max() + 1)

        # set the labels in the same order as the input data
        self.labels_ = concat(new_labels).loc[self.gdf.index]

    def _get_tree(self, training_data, clustering_graph, clustering_kwds):
        '''Carry out AgglomerativeClustering and return the linkage matrix.'''

        clusterer = AgglomerativeClustering(connectivity = clustering_graph,
                                            metric=clustering_kwds['metric'],
                                            linkage=clustering_kwds['linkage'],
                                            compute_full_tree=True,
                                            compute_distances=True)
        model = clusterer.fit(training_data)

        # Create a linkage matrix from a sklearn hierarchical clustering model.
        # by getting the counts for every connection
        counts = zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        return linkage_matrix
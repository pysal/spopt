# ruff: noqa: B006, C408

from sklearn.cluster import AgglomerativeClustering

from ..BaseClass import BaseSpOptHeuristicSolver

import numpy as np
from fast_hdbscan.cluster_trees import (
    cluster_tree_from_condensed_tree,
    condense_tree,
    extract_eom_clusters,
    extract_leaves,
    get_cluster_label_vector,
)


class SA3(BaseSpOptHeuristicSolver):
    """Spatial Adaptive Agglomerative Aggregation. SA^3
    
    Note: This is a conservative clustering in general. Leaf is more conservative than EOM.
    """

    def __init__(self, gdf, w, attrs_name, min_cluster_size=15, eom_clusters=True, clustering_kwds=dict()):

        self.gdf = gdf
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
        
        results = []
        
        for _, group in labels.groupby(labels):
        
            if group.shape[0] <= self.min_cluster_size:
                component_clusters = np.full(group.shape[0], -1)
        
            else:
                component_graph = self.w.subgraph(group.index.values)
                component_data = self.gdf.loc[group.index.values, self.attrs_name]
                component_tree = self._get_tree(component_data, component_graph.transform('B').sparse, self.clustering_kwds)
        
                # # sometimes ward/average linkage breaks the monotonic increase in the MST
                # # if that happens shift all distances by the max drop
                # # need a loop because several connections might be problematic
                problem_idxs = np.where(component_tree[1:, 2] < component_tree[0:-1, 2])[0]
                while problem_idxs.shape[0]:
                    component_tree[problem_idxs + 1, 2] = component_tree[problem_idxs, 2] + .01
                    problem_idxs = np.where(component_tree[1:, 2] < component_tree[0:-1, 2])[0]
                # check if ward tree distances are always increasing
                assert (component_tree[1:, 2] >= component_tree[0:-1, 2]).all()
                
                component_clusters = self._get_clusters(component_tree, 
                                                        self.min_cluster_size, 
                                                        component_data.shape[0], 
                                                        eom_clusters=self.eom_clusters)
            
            results.append(component_clusters)

        ### relabel local clusters - [[0,1,2], [0,1]] to global clusters  - [0, 1, 2, 3, 4]
        new_labels = []
        largest_cluster = 0
        for i, labels_set in enumerate(results):
            new_labels.append(labels + largest_cluster)
            largest_cluster += (labels_set.max() + 1)

        return np.concat(new_labels)

    def _get_tree(self, training_data, clustering_graph, clustering_kwds):
        '''Carry out AgglomerativeClustering and return the linkage matrix.'''

        clusterer = AgglomerativeClustering(connectivity = clustering_graph,
                                            metric=clustering_kwds['metric'],
                                            linkage=clustering_kwds['linkage'],
                                            compute_full_tree=True,
                                            compute_distances=True)
        model = clusterer.fit(training_data)

        # Create a linkage matrix from a sklearn hierarchical clustering model.
        # create the counts of samples under each node
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


    def _get_clusters(self, linkage_matrix, min_cluster_size, n_samples, eom_clusters=True):
        '''Extract hdbscan cluster types from a linkage matrix.'''
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

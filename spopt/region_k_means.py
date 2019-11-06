from .BaseClass import BaseSpOptHeuristicSolver
from .base import (w_to_g, move_ok, ok_moves, region_neighbors, _centroid,
                   _closest, _seeds, is_neighbor)

from collections import defaultdict
import numpy

def region_k_means(X, n_clusters, w):
    """Region-K-means

    K-means with the constraint that each cluster forms a spatially connected component.


    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        The observations to clusters

    n_clusters: int
        The number of clusters to form

    w: spatial weights object


    Returns
    -------

    label: integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the i'th observation is closest to

    centroid: float ndarray with shape (k, n_features)
       Centroids found at the last iteration of region_k_means.

    iters: int
          number of iterations for reassignment phase

    """

    data = X
    a_list = w.to_adjlist(remove_symmetric=False)
    areas = numpy.arange(w.n).astype(int)
    k = n_clusters
    seeds = _seeds(areas, k)

    # initial assignment phase
    label = numpy.array([-1] * w.n).astype(int)
    for i, seed in enumerate(seeds):
        label[seed] = i
    to_assign = areas[label == -1]
    c = 0
    while to_assign.size > 0:
        assignments = defaultdict(list)
        for rid in range(k):
            region = areas[label == rid]
            neighbors = region_neighbors(a_list, region)
            neighbors = [j for j in neighbors if j in to_assign]
            if neighbors:
                d_min = numpy.inf
                centroid = data[region].mean(axis=0)
                for neighbor in neighbors:
                    d = ((data[neighbor] - centroid) ** 2).sum()
                    if d < d_min:
                        idx = neighbor
                        d_min = d
                assignments[idx].append([rid, d_min])
        for key in assignments:
            assignment = assignments[key]
            if len(assignment) == 1:
                r, d = assignment[0]
                label[key] = r
            else:
                d_min = numpy.inf
                for match in assignment:
                    r, d = match
                    if d < d_min:
                        idx = r
                        d_min = d
                label[key] = idx

        to_assign = areas[label == -1]

    # reassignment phase
    changed = []
    g = w_to_g(w)

    iters = 1

    # want to loop this until candidates is empty
    regions = [areas[label == r].tolist() for r in range(k)]
    centroid = _centroid(regions, data)
    closest = numpy.array(_closest(data, centroid))
    candidates = areas[closest != label]
    candidates = ok_moves(candidates, regions, label, closest, g, w, areas)
    while candidates:
        # make moves
        for area in candidates:
            label[area] = closest[area]
        regions = [areas[label == r].tolist() for r in range(k)]
        centroid = _centroid(regions, data)
        closest = numpy.array(_closest(data, centroid))
        candidates = areas[closest != label]
        candidates = ok_moves(candidates, regions, label, closest, g, w, areas)
        iters += 1

    return centroid, label, iters


class RegionKMeansHeuristic(BaseSpOptHeuristicSolver):
    def __init__(self, data, k, w):
        self.data = data
        self.w = w
        self.k = k

    def solve(self):
        centroid, label, iters = region_k_means(self.data, self.k, self.w)
        self.labels_ = label
        self.centroids_ = centroid
        self.iters_ = iters

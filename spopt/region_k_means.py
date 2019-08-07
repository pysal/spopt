from .BaseClass import BaseSpOptHeuristicSolver
from libpysal.weights.util import get_points_array_from_shapefile
from libpysal.io.fileio import FileIO as psopen
from collections import defaultdict
import numpy
import networkx


def w_to_g(w):
    """Get a networkx graph from a PySAL W"""
    g = networkx.Graph()
    for ego, alters in w.neighbors.items():
        for alter in alters:
            g.add_edge(ego, alter)
    return g


def move_ok(area, source, destination, g, w):
    """Check if area can move from source region to destination region"""

    # first check if area has a neighbor in destination
    if not is_neighbor(area, destination, w):
        return False
    # check if moving area would break source connectivity
    new_source = [j for j in source if j != area]
    if networkx.is_connected(g.subgraph(new_source)):
        return True
    else:
        return False


def ok_moves(candidates, regions, labels_, closest, g, w, areas):
    """Check a sequence of candidate moves"""
    keep = []
    for area in candidates:
        source = areas[labels_ == labels_[area]]
        destination = regions[closest[area]]
        if move_ok(area, source, destination, g, w):
            keep.append(area)
    return keep


def region_neighbors(a_list, region):
    """Get neighbors for members of a region"""
    neighbors = a_list[a_list["focal"].isin(region)].neighbor.values
    return [j for j in neighbors if j not in region]


def _centroids(regions, data):
    """Get centroids for all regions"""
    return numpy.array([data[region, :].mean(axis=0) for region in regions])


def _closest(data, centroids):
    """For each row in data, find the closest row in centroids"""
    return [numpy.argmin(((row - centroids) ** 2).sum(axis=1)) for row in data]


def _seeds(areas, k):
    """randomly select k seeds from a sequence of areas"""
    return numpy.random.choice(areas, size=k, replace=False)


def is_neighbor(area, region, w):
    """Check if area is a neighbor of any member of region"""
    neighboring = False
    for member in region:
        if area in w[member]:
            return True
    return False


def region_k_means(X, n_clusters, w):
    """Region-K-means


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
       

    """


class RegionKMeansHeuristic(BaseSpOptHeuristicSolver):
    def __init__(self, data, k, w):
        self.data = data
        self.w = w
        self.k = k

    def solve(self):
        w = self.w
        data = self.data
        a_list = w.to_adjlist(remove_symmetric=False)
        areas = numpy.arange(w.n).astype(int)
        k = self.k
        seeds = _seeds(areas, k)

        # initial assignment phase
        labels_ = numpy.array([-1] * w.n).astype(int)
        for i, seed in enumerate(seeds):
            labels_[seed] = i
        to_assign = areas[labels_ == -1]
        c = 0
        while to_assign.size > 0:
            assignments = defaultdict(list)
            for rid in range(k):
                region = areas[labels_ == rid]
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
                    labels_[key] = r
                else:
                    d_min = numpy.inf
                    for match in assignment:
                        r, d = match
                        if d < d_min:
                            idx = r
                            d_min = d
                    labels_[key] = idx

            to_assign = areas[labels_ == -1]

        # swapping phase
        changed = []
        g = w_to_g(w)

        # want to loop this until candidates is empty
        regions = [areas[labels_ == r].tolist() for r in range(k)]
        centroids = _centroids(regions, data)
        closest = numpy.array(_closest(data, centroids))
        candidates = areas[closest != labels_]
        candidates = ok_moves(candidates, regions, labels_, closest, g, w, areas)

        self.labels_ = labels_

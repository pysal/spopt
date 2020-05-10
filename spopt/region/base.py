"""Base classes for spopt/region"""

from libpysal.io.fileio import FileIO as psopen
import numpy
import networkx


class RegionMixin(object):
    """Mixin class for all region solvers"""

    _solver_type = "regionalizer"

    def solve_assign(self, X, adjacency):
        self.solve(X, adjacency)
        return self.labels_


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


def _centroid(regions, data):
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

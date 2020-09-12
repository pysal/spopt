"""Base classes for spopt/region"""

from libpysal.io.fileio import FileIO as psopen
import numpy
import networkx


class RegionMixin(object):
    """Mixin class for all region solvers."""

    _solver_type = "regionalizer"

    def solve_assign(self, X, adjacency):
        """
        
        Parameters
        ----------
        
        X : 
            ...
        
        adjacency : 
            ...
    
        Returns
        -------
        
        _labels_ : 
            ...
        
        """

        self.solve(X, adjacency)
        _labels_ = self.labels_
        return _labels_


def w_to_g(w):
    """Get a ``networkx`` graph from a PySAL W.
    
    Parameters
    ----------
    
    w : libpysal.weights.W
        ...
    
    Returns
    -------
    
    g : networkx.Graph
        ...
    
    """
    g = networkx.Graph()
    for ego, alters in w.neighbors.items():
        for alter in alters:
            g.add_edge(ego, alter)
    return g


def move_ok(area, source, destination, g, w):
    """Check if area can move from source region to destination region.
    
    Parameters
    ----------
    
    area : 
        ...
    
    source : 
        ...
    
    destination : 
        ...
    
    g : networkx.Graph
        ...
    
    w : libpysal.weights.W
        ...
    
    Returns
    -------
    
    _move_ok_ : bool
        ``True`` if the move is acceptable otherwise ``False``.
    
    """

    _move_ok_ = False

    # first check if area has a neighbor in destination
    if not is_neighbor(area, destination, w):
        return _move_ok_
    # check if moving area would break source connectivity
    new_source = [j for j in source if j != area]
    if networkx.is_connected(g.subgraph(new_source)):
        _move_ok_ = True
        return _move_ok_
    else:
        return _move_ok_


def ok_moves(candidates, regions, labels_, closest, g, w, areas):
    """Check a sequence of candidate moves.
    
    Parameters
    ----------
    
    candidates : 
        ...
    
    regions : 
        ...
    
    labels_ : 
        ...
    
    closest : 
        ...
    
    g : networkx.Graph
        ...
    
    w : libpysal.weights.W
        ...
    
    areas : 
        ...
    
    Returns
    -------
    
    keep : list
        ...
    
    """

    keep = []
    for area in candidates:
        source = areas[labels_ == labels_[area]]
        destination = regions[closest[area]]
        if move_ok(area, source, destination, g, w):
            keep.append(area)
    return keep


def region_neighbors(a_list, region):
    """Get neighbors for members of a region.
    
    Parameters
    ----------
    
    a_list : 
        ...
    
    region : 
        ...
    
    Returns
    -------
    
    _region_neighbors_ : list
        ...
    
    """

    neighbors = a_list[a_list["focal"].isin(region)].neighbor.values
    _region_neighbors_ = [j for j in neighbors if j not in region]
    return _region_neighbors_


def _centroid(regions, data):
    """Get centroids for all regions.
    
    Parameters
    ----------
    
    regions : 
        ...
    
    data : 
        ...
    
    Returns
    -------
    
    _centroid_ : numpy.array
        ...
    
    """

    _centroid_ = numpy.array([data[region, :].mean(axis=0) for region in regions])
    return _centroid_


def _closest(data, centroids):
    """For each row in data, find the closest row in centroids.
    
    Parameters
    ----------
    
    data : 
        ...
    
    centroids : 
        ...
    
    Returns
    -------
    
    _closest_ : list
        ...
    
    """

    _closest_ = [numpy.argmin(((row - centroids) ** 2).sum(axis=1)) for row in data]
    return _closest_


def _seeds(areas, k):
    """Randomly select `k` seeds from a sequence of areas.
    
    Parameters
    ----------
    
    areas : 
        ...
    
    k : int
        The number of desired seeds.
    
    Returns
    -------
    
    _seeds_ : numpy.array
        ...
    
    """

    _seeds_ = numpy.random.choice(areas, size=k, replace=False)
    return _seeds_


def is_neighbor(area, region, w):
    """Check if area is a neighbor of any member of region.
    
    Parameters
    ----------
    
    area : 
        ...
    
    region : 
        ...
    
    w : libpysal.weights.W
        ...
    
    Returns
    -------
    
    neighboring : bool
        ``True`` if area is a neighbor of any member
        of region otherwise ``False``.
    
    """

    neighboring = False
    for member in region:
        if area in w[member]:
            neighboring = True
            return neighboring
    return neighboring

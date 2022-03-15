"""Base classes for spopt/region"""

from libpysal.io.fileio import FileIO as psopen
import libpysal
import numpy
import networkx
from scipy.spatial import KDTree


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




def infeasible_components(gdf, w, threshold_var, threshold):
    """Identify infeasible components.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data

    w : libpysal.weights.W, required
        Weights object created from given data

    attrs_name : list, required
        Strings for attribute names to measure similarity
        (cols of ``geopandas.GeoDataFrame``).

    threshold_var : string, requied
        The name of the spatial extensive attribute variable.

    threshold : {int, float}, required
        The threshold value.

    Returns
    -------
    list of infeasible components
    """
    gdf['_components'] = w.component_labels
    gb = gdf.groupby(by='_components').sum()
    gdf.drop(columns="_components", inplace=True)
    if gb[threshold_var].min() < threshold:
        l = gb[gb[threshold_var]< threshold]
        return l.index.values.tolist()
    return []


def plot_components(gdf, w):
    """Plot to view components of the W for a gdf.

    Parameters
    ----------
    gdf: geopandas.GeoDataframe

    w: libpysal.weights.W defined on gdf

    Returns
    -------
    folium.folium.Map

    """
    cgdf = gdf.copy()
    cgdf['component'] = w.component_labels
    return cgdf.explore(column='component', categorical=True)


def modify_components(gdf, w, threshold_var, threshold, policy='attach'):
    """Modify infeasible components.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data

    w : libpysal.weights.W, required
        Weights object created from given data

    attrs_name : list, required
        Strings for attribute names to measure similarity (cols of
        ``geopandas.GeoDataFrame``).

    threshold_var : string, requied
        The name of the spatial extensive attribute variable.

    threshold : {int, float}, required
        The threshold value.

    policy: str
          'attach' will attach areas of infeasible components to
          nearest neighbor in a feasible component.
          'keep' keeps infeasible components and attempts to solve.


    Returns
    -------
    gdf: geopandas.GeoDataFrame

    w : libpysal.weights.W, required
        Weights object created from given data
    """
    ifcs = infeasible_components(gdf, w, threshold_var, threshold)

    if ifcs == numpy.unique(w.component_labels).tolist():
        raise Exception("No feasible components found in input.")
    policy = policy.lower()
    if not ifcs or policy == 'keep':
        return gdf, w
    elif policy == 'attach':
        ifcas = numpy.where(numpy.isin(w.component_labels, ifcs))[0]
        fcas = numpy.where(~numpy.isin(w.component_labels, ifcs))[0]
        tree = KDTree(list(zip(gdf.iloc[fcas].geometry.centroid.x,
                               gdf.iloc[fcas].geometry.centroid.y)))
        query_pnts = list(zip(gdf.iloc[ifcas].geometry.centroid.x,
                              gdf.iloc[ifcas].geometry.centroid.y))
        dd, jj = tree.query(query_pnts, k=1)
        jj = [fcas[j] for j in jj]
        joins = zip(jj, ifcas)
        original = w.neighbors.copy()

        for left, right in joins:
            original[left].append(right)
            original[right].append(left)
        return gdf, libpysal.weights.W(original)
    elif policy == 'drop':
        keep_ids = numpy.where(~numpy.isin(w.component_labels, ifcs))[0]
        gdf = gdf.iloc[keep_ids]
        cw = libpysal.weights.w_subset(w, keep_ids) 
        new_neigh = {}
        old_new = dict([(o, n) for n, o in enumerate(keep_ids)])
        for old in keep_ids:
            new_key = old_new[old]
            new_neigh[new_key] = [old_new[j] for j in cw.neighbors[old]]
        new_w = libpysal.weights.W(new_neigh)
        gdf.reset_index(inplace=True)
        return gdf, new_w
    else:
        print('undefined components policy')


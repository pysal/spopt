"""Base classes and functions for spopt/region"""

import libpysal
import numpy
import copy
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
        PySAL weights object.

    Returns
    -------

    g : networkx.Graph
        NetworkX representation of ``w``.

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

    area : int, numpy.int64
        Area label.
    source : numpy.array
        Source region labels.
    destination : list
        destination region labels.
    g : networkx.Graph
        NetworkX representation of ``w``.
    w : libpysal.weights.W
        PySAL weights object.

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

    candidates : numpy.array
        Candidate area labels for moves.
    regions : list
        Region labels.
    labels_ : list
        Region membership labels.
    closest : numpy.array
        Closest region labels.
    g : networkx.Graph
        NetworkX representation of ``w``.
    w : libpysal.weights.W
        PySAL weights object.
    areas : numpy.array
        All area labels.

    Returns
    -------

    keep : list
        Area labels where moves are OK.

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

    a_list : pandas.DataFrame
        Adjacency and weights information in dataframe format.
    region : numpy.array
        Single element array for region label.

    Returns
    -------

    _region_neighbors_ : list
        Neighboring region labels of ``region``.

    """

    neighbors = a_list[a_list["focal"].isin(region)].neighbor.values
    _region_neighbors_ = [j for j in neighbors if j not in region]

    return _region_neighbors_


def _centroid(regions, data):
    """Get centroids for all regions.

    Parameters
    ----------

    regions : list
        Region labels.
    data :  numpy.array
        All data coordinates.

    Returns
    -------

    _centroid_ : numpy.array
        Centroid coordinates.

    """

    _centroid_ = numpy.array([data[region, :].mean(axis=0) for region in regions])
    return _centroid_


def _closest(data, centroids):
    """For each row in data, find the closest row in centroids.

    Parameters
    ----------

    data : numpy.array
        All data coordinates.
    centroids : numpy.array
        Centroid coordinates.

    Returns
    -------

    _closest_ : list
        The closest row in ``centroids`` for each row in ``data``.

    """

    _closest_ = [numpy.argmin(((row - centroids) ** 2).sum(axis=1)) for row in data]
    return _closest_


def _seeds(areas, k, seed):
    """Randomly select ``k`` seeds from a sequence of areas.

    Parameters
    ----------

    areas : numpy.array
        Enumeration of ``W`` areas.
    k : int
        The number of desired seeds.
    seed : int
        Random state.

    Returns
    -------

    _seeds_ : numpy.array
        Random labels of clusters to form.

    """

    numpy.random.seed(seed)
    _seeds_ = numpy.random.choice(areas, size=k, replace=False)
    return _seeds_


def is_neighbor(area, region, w):
    """Check if area is a neighbor of any member of region.

    Parameters
    ----------

    area : int, numpy.int64
        The area label.
    region : list
        Region members.
    w : libpysal.weights.W
        PySAL weights object.

    Returns
    -------

    neighboring : bool
        ``True`` if area is a neighbor of any member of region otherwise ``False``.

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

    gdf : geopandas.GeoDataFrame
        Geodataframe containing original data.
    w : libpysal.weights.W
        Weights object created from given data.
    attrs_name : list
        Strings for attribute names to measure similarity
        (cols of ``geopandas.GeoDataFrame``).
    threshold_var : str
        The name of the spatial extensive attribute variable.
    threshold : {int, float}
        The threshold value.

    Returns
    -------

    list
        Infeasible components.

    """

    gdf["_components"] = w.component_labels
    gb = gdf.groupby(by="_components").sum(numeric_only=True)
    gdf.drop(columns="_components", inplace=True)
    if gb[threshold_var].min() < threshold:
        l = gb[gb[threshold_var] < threshold]
        return l.index.values.tolist()
    return []


def plot_components(gdf, w):
    """Plot to view components of the W for a ``gdf``.

    Parameters
    ----------

    gdf : geopandas.GeoDataframe
        Geodataframe of component data.
    w : libpysal.weights.W
        PySAL weights object defined on ``gdf``.

    Returns
    -------

    folium.folium.Map

    """

    cgdf = gdf.copy()
    cgdf["component"] = w.component_labels
    return cgdf.explore(column="component", categorical=True)


def modify_components(gdf, w, threshold_var, threshold, policy="single"):
    """Modify infeasible components.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        Geodataframe containing original data.
    w : libpysal.weights.W
        Weights object created from given data.
    attrs_name : list
        Strings for attribute names to measure similarity (cols of
        ``geopandas.GeoDataFrame``).
    threshold_var : str
        The name of the spatial extensive attribute variable.
    threshold : {int, float}
        The threshold value.
    policy : str (default 'single')
        ``'single'`` will attach an infeasible component to a feasible
        component based on a single join using the minimum nearest
        neighbor distance between areas of infeasible components and
        areas in the largest component. ``'multiple'`` will form a join
        between each area of an infeasible area and its nearest
        neighbor area in a feasible component. ``'keep'`` keeps
        infeasible components and attempts to solve. ``'drop'`` will
        remove the areas from the smallest components and return a ``w``
        defined on the feasible components.

    Returns
    -------

    gdf : geopandas.GeoDataFrame
        Geodataframe containing modified data.
    w : libpysal.weights.W
        Weights object created from given data.

    """

    _policy = policy.lower()
    if _policy not in ["keep", "single", "multiple", "drop"]:
        raise ValueError(f"Unknown `policy`: '{policy}'")

    ifcs = infeasible_components(gdf, w, threshold_var, threshold)
    if ifcs == numpy.unique(w.component_labels).tolist():
        raise ValueError("No feasible components found in input.")

    if not ifcs or _policy == "keep":
        return gdf, w
    elif _policy in ["single", "multiple"]:
        w = form_single_component(gdf, w, linkage=_policy)
        return gdf, w
    else:
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


def form_single_component(gdf, w, linkage="single"):
    """Ensure the connectivity forms a signal connected component.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        Input area data.
    w : libysal.weights.W
        PySAL weights object.
    linkage : str
        `single`: a small component will be joined with the largest
        component by adding a single join based on minimum nearest
        neighbor relation between areas in the small component and
        the largest component. 'multiple': joins are added between each
        area in the small component and their nearest neighbor in the
        largest component.

    Returns
    -------

    w : libpysal.weights.W
        PySAL weights object.

    """

    ll = linkage.lower()
    if ll not in ["single", "multiple"]:
        raise ValueError(f"Unknown `linkage`: '{linkage}'")

    data = numpy.unique(w.component_labels, return_counts=True)
    if len(data[0]) == 1:
        return w
    else:
        # largest component label
        lcl = data[0][numpy.argmax(data[1])]

        # form tree on largest component
        wcl = w.component_labels
        tree = KDTree(
            list(
                zip(
                    gdf.iloc[wcl == lcl].geometry.centroid.x,
                    gdf.iloc[wcl == lcl].geometry.centroid.y,
                )
            )
        )

        # small component labels
        scl = [cl for cl in data[0] if cl != lcl]

        # indices of areas in largest component
        lcas = numpy.where(~numpy.isin(w.component_labels, scl))

        # for all but the largest component, add joins to largest component
        joins = []
        for cl in data[0]:
            if cl == lcl:
                continue
            query_pnts = list(
                zip(
                    gdf.iloc[wcl == cl].geometry.centroid.x,
                    gdf.iloc[wcl == cl].geometry.centroid.y,
                )
            )
            dd, jj = tree.query(query_pnts, k=1)
            clas = numpy.where(numpy.isin(w.component_labels, [cl]))[0]

            # map to idx in gdf
            jj = [lcas[0][j] for j in jj]

            if ll == "single":
                min_idx = numpy.argmin(dd)
                j = jj[min_idx]
                i = clas[min_idx]
                joins.append((i, j))
            else:
                pairs = zip(clas, jj)
                joins.extend(list(pairs))

        neighbors = copy.deepcopy(w.neighbors)
        for join in joins:
            head, tail = join
            neighbors[head].append(tail)
            neighbors[tail].append(head)

        w = libpysal.weights.W(neighbors)

        return w

import geopandas
import numpy
from shapely import MultiPolygon, Point, Polygon
from typing import List, Iterable, Any
import itertools


def simulated_geo_points(
    in_data: geopandas.GeoDataFrame | geopandas.GeoSeries | Polygon | MultiPolygon,
    needed: int = 1,
    seed: int = 0,
) -> geopandas.GeoDataFrame:
    """
    Simulate points within an area.

    This function will wrap a random spatial generator in geopandas once merged.
    See https://github.com/geopandas/geopandas/pull/2363

    Parameters
    ----------

    in_data : geopandas.GeoDataFrame, shapely.{Polygon, MultiPolygon}
        The areal unit in which to generate points.
    needed : int (default 1)
        The number of points to generate.
    seed : int (default 0)
        The random state for number generation.

    Returns
    -------

    sim_pts : geopandas.GeoDataFrame
        The resultant simulated points within ``in_data``.

    Examples
    --------

    >>> import spaghetti
    >>> from spopt.locate.util import simulated_geo_points

    >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
    >>> ntw = spaghetti.Network(in_data=lattice)
    >>> street = spaghetti.element_as_gdf(ntw, arcs=True)

    >>> street_buffered = geopandas.GeoDataFrame(
    ...     geopandas.GeoSeries(street["geometry"].buffer(0.3).unary_union),
    ...     crs=street.crs,
    ...     columns=["geometry"]
    ... )

    >>> points_simulated = simulated_geo_points(street_buffered, needed=10, seed=1)
    >>> type(points_simulated)
    <class 'geopandas.geodataframe.GeoDataFrame'>

    """

    # ensure ``needed`` is a positive whole number
    if isinstance(needed, float) or needed < 1:
        msg = (
            f"Cannot generate {needed} points. "
            "``needed`` must be a positive whole number."
        )
        if needed < 1:
            raise ValueError(msg)
        if not (needed % int(needed)):
            needed = int(needed)
        else:
            raise ValueError(msg)

    # create single areal entity and isolate bounding box
    if isinstance(in_data, geopandas.GeoDataFrame | geopandas.GeoSeries):
        geom = in_data.geometry.unary_union
        xmin, ymin, xmax, ymax = tuple(in_data.total_bounds)
        crs = in_data.crs
    elif isinstance(in_data, Polygon | MultiPolygon):
        geom = in_data
        xmin, ymin, xmax, ymax = in_data.bounds
        crs = None
    else:
        raise ValueError(f"'{type(in_data)}' not valid for ``in_data``.")

    simulated_points_list = []
    simulated_points_all = False
    numpy.random.seed(seed)

    while not simulated_points_all:
        # generate (x,y) coordinates within bounding box of ``geom``
        x = numpy.random.uniform(xmin, xmax, 1)
        y = numpy.random.uniform(ymin, ymax, 1)

        # transform coordinates x, y into `shapely.geometry.Point`
        point = Point(x, y)

        # check if the point is within ``geom`` itself
        if geom.intersects(point):
            simulated_points_list.append(point)

        # check if enough points have been generated
        if len(simulated_points_list) == needed:
            simulated_points_all = True

    sim_pts = geopandas.GeoDataFrame(geometry=simulated_points_list, crs=crs)

    return sim_pts


def rising_combination(
    values: List, start: int = 1, stop: int = None
) -> Iterable[List]:
    """
    Generate combinations of increasing sizes from a list of values.

    Parameters
    ----------
    values : list
        Input list to generate combinations from
    start : int, optional
        Minimum size of combinations (default is 1)
    stop : int or None, optional
        Maximum size of combinations

    Yields
    ------
    List
        Combinations of different sizes
    """
    if stop is None:
        stop = len(values)

    if start < 1:
        raise ValueError("Start must be at least 1")
    if stop > len(values):
        stop = len(values)

    for size in range(start, min(stop + 1, len(values) + 1)):
        yield from map(list, itertools.combinations(values, size))


def rising_combination(
    values: List, start: int = 1, stop: int = None
) -> Iterable[List]:
    """
    Generate combinations of increasing sizes from a list of values.

    Parameters
    ----------
    values : list
        Input list to generate combinations from
    start : int, optional
        Minimum size of combinations (default is 1)
    stop : int or None, optional
        Maximum size of combinations

    Yields
    ------
    List
        Combinations of different sizes
    """
    if stop is None:
        stop = len(values)

    if start < 1:
        raise ValueError("Start must be at least 1")
    if stop > len(values):
        stop = len(values)

    for size in range(start, min(stop + 1, len(values) + 1)):
        yield from map(list, itertools.combinations(values, size))


def compute_facility_usage(
    origin: Any, destination: Any, facility: Any, combination: List
) -> int:
    """
    Compute facility usage coefficient for capacitated model.

    Parameters
    ----------
    origin : Any
        Origin node of the flow
    destination : Any
        Destination node of the flow
    facility : Any
        Facility being evaluated
    combination : List
        List of facilities in the current combination

    Returns
    -------
    int
        Facility usage coefficient (1 if facility is origin/destination, 2 if intermediate, 0 if not in combination)
    """
    if facility not in combination:
        return 0

    if facility == origin or facility == destination:
        return 1

    # Facility is used as an intermediate refueling point
    return 2

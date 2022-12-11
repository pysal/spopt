import geopandas
import numpy
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Union


def simulated_geo_points(
    in_data: Union[geopandas.GeoDataFrame, geopandas.GeoSeries, Polygon, MultiPolygon],
    needed: int = 1,
    seed: int = 0,
) -> geopandas.GeoDataFrame:
    """
    Simulate points within an area.

    This function will wrap a random spatial generator in geopandas once merged.
    See https://github.com/geopandas/geopandas/pull/2363

    Parameters
    ----------

    in_data : geopandas.GeoDataFrame, shapely.geometry.{Polygon, MultiPolygon}
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
    if isinstance(in_data, geopandas.GeoDataFrame) or isinstance(
        in_data, geopandas.GeoSeries
    ):
        geom = in_data.geometry.unary_union
        xmin, ymin, xmax, ymax = tuple(in_data.total_bounds)
        crs = in_data.crs
    elif isinstance(in_data, Polygon) or isinstance(in_data, MultiPolygon):
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

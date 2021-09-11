import numpy
from shapely.geometry import Point
import geopandas


def simulated_geo_points(in_data, needed, seed) -> geopandas.GeoDataFrame:
    """
    Simulate points using a geopandas dataframse with geometry as reference.

    Parameters
    ----------
    in_data: geopandas.GeoDataFrame
        the geodataframe containing the geometries
    needed: int
        how many points to generate
    seed: int
        number to initialize the random number generation

    Returns
    -------
    geopandas.GeoDataFrame

    """

    geoms = in_data.geometry
    area = tuple(
        in_data.total_bounds
    )  # create a polygon with bounds to represent an area
    simulated_points_list = []
    simulated_points_all = False
    numpy.random.seed(seed)
    while simulated_points_all == False:
        x = numpy.random.uniform(
            area[0], area[2], 1
        )  # get coordinates x of area variable
        y = numpy.random.uniform(
            area[1], area[3], 1
        )  # get coordinates y of area variable
        point = Point(x, y)  # transform coordinates x, y into `shapely.geometry.Point`
        if geoms.intersects(point)[0]:  # check if the point belong to the network
            simulated_points_list.append(point)
        if (
            len(simulated_points_list) == needed
        ):  # check if the length of array of points simulated
            # contains the number of points needed
            simulated_points_all = True
    sim_pts = geopandas.GeoDataFrame(
        simulated_points_list, columns=["geometry"], crs=in_data.crs
    )  # transform the points array into geodataframe

    return sim_pts
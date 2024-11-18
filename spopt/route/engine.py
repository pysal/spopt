try:
    import osrm
    has_bindings = True
except (ImportError,ModuleNotFoundError) as e:
    has_bindings = False
import os
import numpy
import requests
import warnings
import geopandas
import shapely
from sklearn import metrics

# TODO: needs to be configurable by site
_OSRM_DATABASE_FILE = ""

def build_route_table(demand_sites, candidate_depots, cost='distance', http=not has_bindings, database_path=_OSRM_DATABASE_FILE, port=5000):
    """
    Build a route table using OSRM, either over http or over py-osrm bindings
    """
    if isinstance(demand_sites, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        demand_sites = demand_sites.geometry.get_coordinates().values
    if isinstance(candidate_depots, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        candidate_depots = candidate_depots.geometry.get_coordinates().values
    if cost not in ("distance", "duration", "both"):
        raise ValueError(f"cost option '{cost}' not one of the supported options, ('distance', 'duration', 'both')")
    if http:
        try: 
            distances, durations = _build_route_table_http(demand_sites, candidate_depots, cost=cost, port=port)
        except (requests.ConnectionError, requests.JSONDecodeError):
            warnings.warn(
                "Failed to connect to routing engine... using haversine distance"
                " and (d/500)**.75 for durations"
            )
            distances = metrics.pairwise_distances(
                    numpy.fliplr(numpy.deg2rad(demand_sites)), 
                    numpy.fliplr(numpy.deg2rad(candidate_depots)),
                    metric="haversine"
                ) * 6371000
            durations = numpy.ceil((distances / 10) ** .75)
    else:
        distances, durations = _build_route_table_pyosrm(
            demand_sites, candidate_depots, database_path=database_path 
        )
    for D in (distances, durations):
        if D is None:
            continue
        n_row, n_col = D.shape
        assert n_row == len(candidate_depots)
        assert n_col == len(demand_sites)
        no_route_available = numpy.isnan(D)
        D[no_route_available] = D[~no_route_available].sum()
    if cost == 'distance':
        return distances
    elif cost == 'duration':
        return durations
    elif cost == 'both':
        return distances, durations

def build_specific_route(waypoints, port=5000, http=not has_bindings, return_durations=True, database_path=_OSRM_DATABASE_FILE):
    """
    Build a route over the road network from each waypoint to each other waypoint. If the routing engine is not found, this builds straight-line 
    routes, and measures their duration as a nonlinear function of the 
    haversine distance between input points. 
    """
    if isinstance(waypoints, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        waypoints = waypoints.geometry.get_coordinates().values
    if http:
        try:
            out = _build_specific_route_http(waypoints, port=port, return_durations=return_durations)
        except (requests.ConnectionError, requests.JSONDecodeError):
            warnings.warn(
                "Failed to connect to routing engine... constructed routes"
                " will be straight lines and may not follow the road network."
            )
            route = shapely.LineString(waypoints)
            prep_points = numpy.fliplr(numpy.deg2rad(waypoints))
            durations = [
                (metrics.pairwise.haversine_distances([prep_points[i]], [prep_points[i+1]]) 
                 * 637000 / 10)**.75
                for i in range(len(prep_points)-1)
            ]
            out = (route, durations) if return_durations else route
    else:
        route = _build_specific_route_pyosrm(waypoints, database_path=database_path, return_durations=return_durations)
    if return_durations:
        route, durations = out
        return route, durations
    else:
        route = out
        return route

def _build_specific_route_http(waypoints, return_durations=True, port=5000):
    
    # TODO: needs to be configurable by site
    baseurl = f"http://127.0.0.1:{int(port)}/route/v1/driving/"

    point_string = ";".join(
        map(
            lambda x: "{},{}".format(*x),
            waypoints,
        )
    )

    request_url = (
        baseurl 
        + point_string
        + "?"
        + "steps=true"
        + "&"
        + f"geometries=geojson"
        + "&"
        + "annotations=true"
    )
    routes = requests.get(request_url).json()['routes']
    assert len(routes) == 1
    route = routes[0]
    #sub_coordinates = numpy.empty(shape=(0,2))
    route_shape = shapely.geometry.shape(route['geometry'])
    leg_durations = numpy.array([leg['duration'] for leg in route['legs']])
    """
    for leg_i, leg in enumerate(route['legs']):
        durations[i] = leg['duration']
        for steps in leg['steps']:
            assert steps['geometry']['type'] == "LineString"
            sub_coordinates = numpy.row_stack((sub_coordinates, 
                             numpy.asarray(steps['geometry']['coordinates'])[:-1]
            ))
    """
    #route_shape = shapely.LineString(sub_coordinates)
    numpy.testing.assert_array_equal(
        shapely.get_num_geometries(route_shape),
        numpy.ones((len(waypoints),))
    )
    if return_durations:
        return route_shape, leg_durations
    else:
        return route_shape

def _build_specific_route_pyosrm(waypoints, database_path=_OSRM_DATABASE_FILE, return_durations=False):
    raise NotImplementedError()

def _build_route_table_http(demand_sites, candidate_depots, cost='distance', port=5000):
    """
    Build a route table using the http interface to the OSRM engine
    """
    request_url = _create_route_request(demand_sites, candidate_depots, cost=cost, port=port)
    request = requests.get(request_url)
    content = request.json()
    if cost == 'distance':
        D = numpy.asarray(content["distances"]).astype(float)
        output = (D,None)
    elif cost == 'duration':
        D = numpy.asarray(content["durations"]).astype(float)
        output = (None,D)
    elif cost == 'both':
        distances = numpy.asarray(content["distances"]).astype(float)
        durations = numpy.asarray(content["durations"]).astype(float)
        output = (distances, durations)
    else:
        raise ValueError(f"cost option '{cost}' not one of the supported options, ('distance', 'duration', 'both')")
    return output


def _create_route_request(demand_sites, candidate_depots, cost='distance', port=5000):
    point_string = ";".join(
        map(
            lambda x: "{},{}".format(*x),
            numpy.row_stack((candidate_depots, demand_sites)),
        )
    )
    n_demands = len(demand_sites)
    n_supplys = len(candidate_depots)
    source_string = "sources=" + ";".join(numpy.arange(n_supplys).astype(str))
    destination_string = "destinations=" + ";".join(
        numpy.arange(n_supplys, n_demands + n_supplys).astype(str)
    )
    # TODO: needs to be configurable by site
    baseurl = f"http://127.0.0.1:{int(port)}/table/v1/driving/"
    if cost=='distance':
        annotation = "&annotations=distance"
    elif cost=='duration':
        annotation = "&annotations=duration"
    elif cost=='both':
        annotation = "&annotations=duration,distance"
    else:
        annotation = ""

    request_url = (
        baseurl
        + point_string
        + "?"
        + source_string
        + "&"
        + destination_string
        + annotation
        + "&exclude=ferry"
    )
    return request_url


def _build_route_table_pyosrm(demand_sites, candidate_depots, database_path=_OSRM_DATABASE_FILE): 
    """
    build a route table using py-osrm
    https://github.com/gis-ops/py-osrm
    """
    engine = osrm.OSRM(
        storage_config=database_path, 
        use_shared_memory=False
        )
    n_demands = len(demand_sites)
    n_supplys = len(candidate_depots)
    query_params = osrm.TableParameters(  # noqa: F821
        coordinates=[
            (float(lon), float(lat)) 
            for (lon, lat)
            in numpy.row_stack((demand_sites, candidate_depots))
        ],
        sources=list(numpy.arange(n_demands)),
        destinations=list(numpy.arange(n_demands, n_demands + n_supplys)),
        annotations=["distance"],
    )
    res = engine.Table(query_params)
    return numpy.asarray(res["distances"]).astype(float).T

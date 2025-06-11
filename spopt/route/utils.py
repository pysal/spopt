import numpy
import pandas
import routing
import copy
import geopandas
import shapely
import randomname
MIDNIGHT = pandas.to_datetime(
    "2030-01-02 00:00:00", format='%Y-%m-%d %H:%M:%S'
)

def integralize_time_windows(windows, freq=pandas.Timedelta(minutes=1)):
    """
    Convert time windows to a count of time units since midnight. 

    Arguments
    ---------
    windows :   pandas.DataFrame
        a dataframe containing the time windows
    freq    :   pandas.TimeDelta 
        the base time unit to use for counting. Defaults to one minute. 
    """
    timeblocks = pandas.Series(pandas.date_range(
        start="2030-01-02 00:00:00",
        end="2030-01-02 23:59:59",
        freq=freq
    ))
    output = []
    for col in windows.columns:
        dt = pandas.to_datetime(windows[col])
        itime = dt.apply(
            lambda t: _integralize_time(t, timeblocks)
            )
        output.append(itime)
    return numpy.column_stack(output)

def _integralize_time(t, timeblocks):
    """
    Find the time block in timeblocks to which time t corresponds. 
    If time t is null, return -1, which is assumed to be handled elsewhere. 
    If time t is is not Null, then the time block into which t falls is 
    returned. 
    If time t does not fall into any block, then it's assumed to fall into 
    a block past the final block. 
    """
    if pandas.isnull(t):
        return -1
    in_blocks = timeblocks.gt(t, fill_value=pandas.NaT)
    if in_blocks.any():
        return in_blocks.argmax(skipna=False)
    return len(timeblocks)

def integralize_lonlat(lonlat, offset=True):
    """
    Convert the lonlat values to integer meters relative to a UTM projection.

    Arguments
    ---------
    lonlat  :   numpy.ndarray or geopandas.GeoSeries/geopandas.GeoDataFrame
        input geometries or coordinate values to convert to integer positions
    offset  : bool
        whether or not to remove the false northing/easting on the UTM
        reference frame, making the bottom left of the reference frame correspond to location 0,0. This is done by default.
    
    Returns
    -------
    numpy.ndarray of integer coordinate locations to the nearest UTM-meter. 
    """
    if isinstance(lonlat, numpy.ndarray):
        lon, lat = lonlat.T
        lonlat = geopandas.GeoSeries(geopandas.points_from_xy(
            x=lon, 
            y=lat,
            crs='EPSG:4326'
            )
        )
    elif isinstance(lonlat, (geopandas.GeoSeries,geopandas.GeoDataFrame)):
        if lonlat.crs is None:
            lonlat = lonlat.set_crs("epsg:4326")
    utm = lonlat.estimate_utm_crs()
    geoms = lonlat.to_crs(utm).geometry
    raw_coords = numpy.column_stack((geoms.x, geoms.y))
    if not offset:
        raw_coords -= geoms.total_bounds[:2]
    return raw_coords.round(0).astype(int)

def calculate_minimum_unit(vector):
    """
    Calculate the smallest positive value in the vector. 
    Used throughout to estimate the integer "unit" for some of the
    integralize_* functions.
    """
    min_unit = vector[vector>0].min()
    return min_unit

def integralize_demand(demands, unit=None):
    """
    Convert the demand values to integer units in terms of the minimum
    nonzero demand. This converts the demand units to a count of
    minimum nonzero demand values required to satisfy the client. Thus, 
    integralize(demands, unit) * unit will always be at least as big as 
    demands themselves, and may be larger by at most "unit"

    Arguments
    ---------
    demands : numpy.ndarray
        demands used in the problem that must be integralized to a count. 
    unit  :   float
        demand value to use as the unit count for the demands. 
    """
    if unit is None:
        unit = calculate_minimum_unit(demands)
    return numpy.ceil(demands/unit).astype(int), unit

def routes_and_stops(
    solution, 
    model, 
    target_geoms, 
    depot_location,
    cost_unit=1e-4
    ):
    """
    Calculate route geometries and stop etas/waypoint numbers from an input
    solution.

    Arguments
    ---------
    solution    :   pyvrp.Solution
        routing solution from running the pyvrp solver that describes the
        best available routes to satisfy the constraints and specifications 
        recorded in the `model`
    model   :   pyvrp.Model
        the model reflecting the problem specification that has been solved
        and recored in `solution`
    target_geoms    :   geopandas.GeoSeries/geopandas.GeoDataFrame
        the real-world longitude and latitude that correspond to the clients 
        recorded in the model. This should *not* include the depot location
        which is provided as a separate argument, unless the depot is also 
        located at a client. 
    depot_location  :   tuple
        the longitude and latitude values that correspond to the location
        of the delivery depot
    
    Returns
    -------
    two dataframes containing the routes and stops. the routes dataframe 
    will have one row per route, while the stops dataframe will be the same length 
    as the target_geoms input
    """
    assert solution.is_feasible(), "solution is infeasible, routes cannot be used."
    assert solution.is_complete(), "solution does not visit all required clients, routes cannot be used."
    n_routes = solution.num_routes()
    route_names = list(randomname.sample("adj/", "names/surnames/french",
        n=n_routes
    ))

    problem_data = model.data()

    # problem assumes all trucks have the same departure time
    # problem assumes that this is in minutes since 00:00:00
        
    route_lut = dict(zip(route_names, solution.routes()))
    stops = [
        (route_name, r.visits()) 
        for route_name, r in route_lut.items()
    ]

    stops = pandas.DataFrame(
        stops
    ).rename(
        columns={0:"route_name", 1:"stop_idx"}
    ).set_index("route_name")
    
    # calculate visit time, 
    # distances and durations are assumed constant over 
    # vehicle type
    duration_matrix, = problem_data.duration_matrices()
    distance_matrix, = problem_data.distance_matrices()
    # TODO: would it be helpful to have the running capacity? 
    def timedelta_from_visits(
        route, 
        duration_matrix=duration_matrix, 
        locations=model.locations
    ):  
        """
        This is a private function to estimate the time changes
        that evolve over a route using the model specific information,
        rather than using the osrm-provided durations on demand. 
        This is to account for any waiting that occurs at the stops. 
        """
        full_visits = [0, *route.visits(), 0]
        arrival_minutes = [route.start_time()]
        for stop_number, stop_idx in enumerate(full_visits[:-1]):
            next_stop_idx = full_visits[stop_number + 1]
            travel_duration = duration_matrix[stop_idx, next_stop_idx]
            # if service duration is not recorded, we assume
            # there is no service time (like, for a waypoint)
            service_duration = getattr(
                locations[stop_idx], "service_duration", 0
            )
            # once you're at stop_idx, you spend service_duration
            # there, and then spend travel_duration to get to the
            # next spot. So, the deltas should be 
            # [0, service_duration[1] + travel_duration[0,1], ...]
            # since the depot has service duration 0
            arrival_time = arrival_minutes[stop_number] + service_duration + travel_duration
            # if you arrive at the target before it's open, then you have to wait
            arrival_time = numpy.maximum(
                getattr(
                    locations[stop_idx],
                    "tw_early",
                    -numpy.inf
                ), arrival_time
            )
            arrival_minutes.append(arrival_time)
        tds = pandas.to_timedelta(arrival_minutes, unit='minutes')
        return tds
    
    stops['eta'] = pandas.Series(
        {name:timedelta_from_visits(r)[1:-1] + MIDNIGHT
        for name,r in route_lut.items()}
    )
    stops['stop_number'] = stops.stop_idx.apply(lambda x: numpy.arange(len(x))+1)
    
    big_stops = stops.explode(
        ["stop_idx", "stop_number", "eta"]
    )
    big_stops['target_uid'] = [
        model.locations[s].name for s in big_stops.stop_idx
    ]
    big_stops['stop_number'] = big_stops.groupby("route_name").cumcount().astype(int) + 1

    stop_output = target_geoms.copy(deep=True)
    stop_output = big_stops.reset_index().merge(
        target_geoms, left_on='target_uid', right_index=True,
        how='right'
        )
    stop_output['route_name'] = stop_output.route_name.fillna("unassigned")
    stop_output['stop_number'] = stop_output.stop_number.fillna(-1)
    stop_output = stop_output.sort_values(["route_name","stop_number"])
    stop_output = geopandas.GeoDataFrame(
        stop_output, 
        geometry='geometry', 
        crs=target_geoms.crs
    )

    route_data = []

    for name, group in stop_output.groupby("route_name"):
        route_obj = route_lut[name]
        group = group.sort_values("stop_number")
        coordinates = shapely.get_coordinates(group.geometry)
        shape, durations = routing.build_specific_route(
            numpy.vstack(
                (
                depot_location,
                coordinates, 
                depot_location
                )
            )
        )
        route_truck_type = route_obj.vehicle_type()
        truck_obj = model.vehicle_types[route_truck_type]
        deptime, rettime = pandas.to_timedelta([
                route_obj.start_time(),
                route_obj.end_time()
                ], unit="minutes"
                ) + MIDNIGHT

        route_data.append((
            name,
            truck_obj.name,
            route_obj.duration(),
            route_obj.distance(),
            route_obj.distance_cost() * cost_unit,
            route_obj.duration_cost() * cost_unit,
            truck_obj.fixed_cost * cost_unit,
            ( route_obj.distance_cost() 
             + route_obj.duration_cost() 
             + truck_obj.fixed_cost
            ) * cost_unit,
            deptime,
            rettime,
            round(route_obj.duration() / truck_obj.max_duration * 100, 2),
            round(route_obj.delivery() / truck_obj.capacity * 100, 2),
            round(route_obj.distance() / truck_obj.max_distance * 100, 2),
            shape
        ))
    
    route_output = geopandas.GeoDataFrame(
        pandas.DataFrame(
            route_data,
            columns = [
                'route_name',
                'truck_type',
                'duration_min',
                'distance_m',
                'fuel_cost_€',
                'labor_cost_€',
                'truck_cost_€',
                'total_cost_€',
                'departure',
                'arrival',
                'utilization_time',
                'utilization_load',
                'utilization_rangelimit',
                'geometry'
            ]
        ),
        geometry='geometry', 
        crs=target_geoms.crs
        )
  
    return route_output, stop_output

def route_webmap(
    problem, model, locs, depot_location, return_dataframes=False
    ):
    """
    Create a webmap from input data. 

    Arguments
    ---------
    solution    :   pyvrp.Solution
        routing solution from running the pyvrp solver that describes the
        best available routes to satisfy the constraints and specifications 
        recorded in the `model`
    model   :   pyvrp.Model
        the model reflecting the problem specification that has been solved
        and recored in `solution`
    target_geoms    :   geopandas.GeoSeries/geopandas.GeoDataFrame
        the real-world longitude and latitude that correspond to the clients 
        recorded in the model. This should *not* include the depot location
        which is provided as a separate argument, unless the depot is also 
        located at a client. 
    depot_location  :   tuple
        the longitude and latitude values that correspond to the location
        of the delivery depot
    return_dataframes   :   bool
        whether or not to return the routes and stops dataframes as well
        as the route map. If True, then the output is returned as a tuple
        containing (map, routes, stops). Otherwise, the output is just the map. 
    
    Returns
    -------
    a folium.Map, and (if return_dataframes==True), two dataframes containing
    the routes and stops. the routes dataframe will have one row per route, 
    while the stops dataframe will be the same length as the target_geoms input
    """
    routes, stops = routes_and_stops(
        problem, model, locs, depot_location
    )
    m = routes.sort_values("route_name").explore(
        "route_name", 
        categorical=True, 
        tiles="CartoDB positron"
    )
    stops_for_map = stops.copy()
    stops_for_map['eta'] = stops.eta.astype(str)
    stops_for_map[
        ["target_uid", 'stop_number', 'route_name', 'eta', 
        '1rst_opening_hours', '1rst_closing_hours', 
        '2nd_opening_hours', '2nd_closing_hours',
        'sum_delivered_volume', 'geometry']
    ].explore("route_name",
        m=m, legend=False,
        style_kwds=dict(color='black', radius=3, weight=1.5)
    )
    geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(
            x=[depot_location[0]], y=[depot_location[1]],
            crs="epsg:4326"
        )
    ).explore(m=m, color="black", marker_type="marker")
    if return_dataframes:
        return m, routes, stops
    return m

def _to_dict(input, indices=None):
    """
    construct a dictionary from an iterable for use in networkx updates
    """
    if isinstance(input, dict):
        return input
    if indices is None:
        indices = range(len(input))
    try:
        return input.to_dict()
    except AttributeError:
        return dict(zip(indices, input))
    
def restrict_by_zone(
        targets, 
        restrictions, 
        distances, 
        depot_to_points, 
        points_to_depot, 
        penalty_value=None
        ):
    """
    the restriction dataframe must be indexed by
    the truck type that cannot go into that zone. 
    So, if truck type 2 cannot go into restriction
    zone 7, the seventh row should have "2" as its index,
    *not* "large petrol".
    """
    distances = copy.deepcopy(distances)
    depot_to_points = copy.deepcopy(depot_to_points)
    points_to_depot = copy.deepcopy(points_to_depot)
    try:
        restrictions.index.astype(int)
        last_index = restrictions.index.max()
        distances[:,:,last_index]
    except (ValueError, IndexError):
        raise ValueError(
            "the restrictions dataframe must be indexed by the integer"
            " the restricted truck. So, if truck type 2 cannot go into"
            " restriction zone 7, then the seventh row of `restrictions`"
            " must have index 2, and distances[2] should be the cost/distance"
            " matrix for that truck type."
            )
    if penalty_value is None: 
        # works for any number of distance matrices
        penalty_value =  numpy.sum(distances) 

    rzone_i, target_i = targets.sindex.query(
        restrictions.geometry,
        predicate='intersects'
    )

    for i, truck_i in enumerate(restrictions.index):
        mask = (rzone_i == i)
        if mask.sum() > 0:
            # this is the entire row space; can't enter
            distances[target_i[mask], :, truck_i] = penalty_value
            # this is the entire column space; can't leave
            distances[:, target_i[mask], truck_i] = penalty_value
            # shouldn't start there
            depot_to_points[target_i[mask], truck_i] = penalty_value
            # or return from there
            points_to_depot[target_i[mask], truck_i] = penalty_value
        else:
            continue
    
    return distances, depot_to_points, points_to_depot




def build_clients(
    model,
    xy_ints,
    demands,
    time_windows,
    names,
    service_durations,
    ):
    """
    Legacy function to construct a collection of clients 
    from an input set of demands, time windows, names, and service times.
    """
    time_ints = integralize_time_windows(time_windows)
    demand_ints, demand_units = integralize_demand(
        demands
    )
    service_timeblock_durations = numpy.ceil(
            pandas.to_timedelta(
                service_durations
            ).dt.total_seconds().values / 60 
        )
    clients = []
    for i,name in enumerate(names):
        x, y = xy_ints[i]
        x, y = x.item(), y.item()
        delivery = demand_ints[i].item()
        service_duration=service_timeblock_durations[i].item()
        tws = time_ints[i]
        if (tws[1]>tws[2]): # no lunchbreak
            client_ = model.add_client(
                x=x,
                y=y,
                delivery=delivery,
                tw_early=tws[0].item(),
                tw_late=tws[1].item(),
                name=name,
                service_duration = service_duration
            )
            clients.append(client_)
        else:
            cg_ = model.add_client_group()
            c1 = model.add_client(
                x=x,
                y=y,
                delivery=delivery,
                tw_early=tws[0].item(),
                tw_late=tws[1].item(),
                name=name,
                required=False,
                service_duration = service_duration,
                group=cg_
            )
            c2 = model.add_client(
                x=x,
                y=y,
                delivery=delivery,
                tw_early=tws[2].item(),
                tw_late=tws[3].item(),
                name=name,
                required=False,
                service_duration = service_duration,
                group=cg_
            )
            clients.extend([c1, c2])
    return clients
import geopandas
import numpy
import pandas
import shapely

import pyvrp
from . import engine
from . import utils

class SpecificationError(Exception):
    pass

_MAX_INT = numpy.iinfo(numpy.int64).max

class LastMile:
    def __init__(
        self,
        depot_location=None,
        depot_open=pandas.to_datetime("2030-01-02 07:45:00"),
        depot_close=None,
        depot_name=None,
        cost_unit=1e-4,
    ):
        """
        Initialize a LastMile problem. 

        Arguments
        ---------
        depot_location  :   tuple
            longitude and latitude of depot for whom routes must be drawn. 
        depot_open  :   pandas.Datetime 
            The time from which trucks can start leaving the depot. 
        depot_close  :   pandas.Datetime 
            The time by which trucks must return to the depot.
        depot_name  :   str
            the name of the depot facility being sited
        cost_unit   :    float
            the fraction of cost values to round to. This is set to hundreths of 
            a cent by default (1e-4), assuming inputted costs are in typical 
            decimal notation for euros and cents. 
        """
        self.model = pyvrp.Model()
        self.depot_location = depot_location
        self.depot_open = depot_open
        self.depot_close = depot_close
        self.depot_name = depot_name
        self.cost_unit = cost_unit

    def add_truck_type(
        self,
        name=None,
        n_truck=None,
        capacity=7,
        time_windows=None,
        fixed_cost=None,
        cost_per_meter=None,
        cost_per_minute=30 / 60,
        max_duration=pandas.Timedelta(hours=8, minutes=00),
        max_distance=None,
    ):
        """
        Add a single vehicle type to the LastMile problem. Must have
        added clients first before setting up trucks, since the truck capacity 
        will be re-scaled to provide the count of minimum delivery values the
        truck can carry. 

        Parameters
        ----------
        name    :   str
            name of the truck type being added  
        n_truck :   int
            how many of this truck type are available to use
        capacity    : float
            how big the truck is. Must be measured in the same units as
            the client demand input
        time_windows    : pandas.DataFrame
            data frame containing open and close times for this truck type's
            routes. If not provided, these are set to the depot open and close
            times. But, you may pass a custom value here if you want to, say, 
            force big noisy trucks to come back to the depot early
        fixed_cost  :   float
            the fixed cost per day of using a truck of this type
        cost_per_meter  :   float
            the variable cost per meter distance of using this truck
        cost_per_minute :   
            the variable cost per minute time of using this truck
        max_duration    : pandas.Timedelta
            the total allowed length of the route
        max_distance    : float
            the total allowed distance (in meters) of the route. 
        
        Returns
        -------
        this LastMile object with these new truck types
        """
        if not hasattr(self, "demand_unit_"):
            raise SpecificationError(
                "You must set the clients before adding truck types."
            )
        if time_windows is None:
            time_windows = pandas.Series([self.depot_open, self.depot_close], index=['open_1', 'close_1']).to_frame(self.depot_name).T
        depot_ints = utils.integralize_time_windows(time_windows)
        if name is None:
            if hasattr(self, "trucks_"):
                name = str(len(self.trucks_) + 1)
            else:
                name = "0"
        v_ = self.model.add_vehicle_type(
            name=str(name),
            num_available=int(n_truck),
            capacity=int(capacity / self.demand_unit_),
            fixed_cost=int(fixed_cost / self.cost_unit),
            tw_early=depot_ints[0,0],
            tw_late=depot_ints[0,1],
            max_duration=int(max_duration.total_seconds() / 60),
            max_distance=max_distance if max_distance is not None else _MAX_INT,
            unit_distance_cost=int(cost_per_meter / self.cost_unit),
            unit_duration_cost=int(cost_per_minute / self.cost_unit),
        )
        if hasattr(self, "trucks_"):
            self.trucks_.append(v_)
        else:
            self.trucks_ = [v_]
        return self
    
    def add_trucks_from_frame(
            self,
            truck_frame,
            n_trucks=None
    ):
        """
        Add a single vehicle type to the LastMile problem. Must have
        added clients first before setting up trucks, since the truck capacity 
        will be re-scaled to provide the count of minimum delivery values the
        truck can carry. 

        New truck types will be added to the self.trucks_ list.

        Parameters
        ----------
        name    :   str
            name of the truck type being added  
        n_trucks :   int
            how many trucks in total are allowed across all the inputted truck types. 
            If this option is provided, then the dataframe's n_truck column will govern
            the *fraction* of trucks for each type, rounded down so that trucks do not
            exceed the n_trucks limit. The fraction is calculated as 
            `truck_frame.n_truck / truck_frame.n_truck.sum()`
        capacity    : float
            how big the truck is. Must be measured in the same units as
            the client demand input
        time_windows    : pandas.DataFrame
            data frame containing open and close times for this truck type's
            routes. If not provided, these are set to the depot open and close
            times. But, you may pass a custom value here if you want to, say, 
            force big noisy trucks to come back to the depot early
        fixed_cost  :   float
            the fixed cost per day of using a truck of this type
        cost_per_meter  :   float
            the variable cost per meter distance of using this truck
        cost_per_minute :   
            the variable cost per minute time of using this truck
        max_duration    : pandas.Timedelta
            the total allowed length of the route
        max_distance    : float
            the total allowed distance (in meters) of the route. 
        
        Returns
        -------
        A LastMile() object modified in place to add a new truck type
        """
        if n_trucks is not None:
            raise NotImplementedError("will not yet calculate truck fractions")
        keep_cols = truck_frame.columns.isin(
            ['name',
             'n_truck',
             'capacity',
             'time_windows',
             'fixed_cost',
             'cost_per_meter',
             'cost_per_minute',
             'max_duration',
             'max_distance']
            )
        for name, row in truck_frame.loc[:,keep_cols].iterrows():
            truck_spec = row.to_dict()
            if 'name' not in truck_spec.keys():
                truck_spec['name'] = name
            self.add_truck_type(**truck_spec)
        return self

    def add_clients(
        self,
        locations,
        delivery=None,
        pickup=None,
        time_windows=None,
        names=None,
        service_times=None,
    ):
        """
        Add delivery targets to the model

        Parameters
        ----------
        locations   : geopandas.GeoSeries/geopandas.GeoDataFrame   
            the locations to add to the problem
        delivery  :   numpy.ndarray
            the demand values used to calculate load to be delivered to clients
        pickup  :   numpy.ndarray
            the demand values used to calculate load to be picked up from clients along a route
        time_windows    :   pandas.DataFrame
            open and close windows for each client. If every client has an open and a close, then 
            this dataframe should have two columns. If some clients have two open periods,
            then this dataframe should have four columns, with open and close times interleaved.
        names   :   numpy.ndarray
            names to use for each client. this should serve as a unique ID for the client. 
        service_times   :   numpy.ndarray
            the amount of time it takes to service the client, either as a string that is passed
            directly to pandas.to_timedelta() or an existing iterable of timedelta objects. 
            
        Returns
        -------
        this LastMile() object with clients set to the clients_ attribute, 
        a depot objecti assigned as the depot_ attribute, and the unit of
        demand assigned to the demand_unit_ attribute. 
        """
        coords = shapely.get_coordinates(locations.geometry)
        all_coords = numpy.row_stack((self.depot_location, coords))
        n_clients = coords.shape[0]

        locints = utils.integralize_lonlat(all_coords)
        depot_xy, client_xy = locints[0, :], locints[1:, :]

        self.depot_ = self.model.add_depot(
            x=depot_xy[0], y=depot_xy[1], name=self.depot_name
        )
        if time_windows is None:
            time_windows = pandas.DataFrame.from_dict(
                {"open_1":[self.depot_open]*len(locations.geometry),
                 "close_1": [self.depot_close]*len(locations.geometry)
                 },
            )
            time_windows.index = locations.geometry.index
        if names is None:
            names = locations.geometry.index
        if service_times is None:
            service_times = pandas.Series(numpy.zeros((n_clients,)), index=names).astype(int)
        time_ints = utils.integralize_time_windows(time_windows)
        if (delivery is None) | (pickup is None):
            if (delivery is None):
                delivery = pandas.Series(numpy.zeros((n_clients,)), index=names).astype(int)
            if (pickup is None):
                pickup = pandas.Series(numpy.zeros((n_clients,)), index=names).astype(int)
        self.demand_unit_ = utils.calculate_minimum_unit(numpy.hstack((delivery, pickup)))
        delivery_ints,_ = utils.integralize_demand(delivery, unit=self.demand_unit_)
        pickup_ints,_ = utils.integralize_demand(pickup, unit=self.demand_unit_)
        
        service_timeblocks = numpy.ceil(
            pandas.to_timedelta(service_times).dt.total_seconds().values / 60
        )
        clients = []

        for i, name in enumerate(names):
            x, y = client_xy[i]
            x, y = x.item(), y.item()
            delivery = delivery_ints.iloc[i].item()
            pickup = pickup_ints.iloc[i].item()
            service_duration = service_timeblocks[i].item()
            tws = time_ints[i]
            if len(tws) == 2:
                lunchbreak=False
            else:
                lunchbreak = tws[1] > tws[2]
            if not lunchbreak:  # no lunchbreak
                client_ = self.model.add_client(
                    x=x,
                    y=y,
                    delivery=delivery,
                    pickup=pickup,
                    tw_early=tws[0].item(),
                    tw_late=tws[1].item(),
                    name=str(name),
                    service_duration=service_duration,
                )
                clients.append(client_)
            else:
                cg_ = self.model.add_client_group()
                c1 = self.model.add_client(
                    x=x,
                    y=y,
                    delivery=delivery,
                    pickup=pickup,
                    tw_early=tws[0].item(),
                    tw_late=tws[1].item(),
                    name=name,
                    required=False,
                    service_duration=service_duration,
                    group=cg_,
                )
                c2 = self.model.add_client(
                    x=x,
                    y=y,
                    delivery=delivery,
                    pickup=pickup,
                    tw_early=tws[2].item(),
                    tw_late=tws[3].item(),
                    name=name,
                    required=False,
                    service_duration=service_duration,
                    group=cg_,
                )
                clients.extend([c1, c2])
        timedict = dict(open_1=time_windows.iloc[:, 0], close_1=time_windows.iloc[:, 1])
        if time_windows.shape[1] == 4:
            timedict.update(dict(open_2=time_windows.iloc[:, 2], close_2=time_windows.iloc[:, 3]))
        self.clients_ = geopandas.GeoDataFrame(
            pandas.DataFrame.from_dict(
                dict(delivery=delivery, pickup=pickup, service_time=service_times)
            ).assign(**timedict),
            index=names,
            geometry=locations.geometry,
            crs=locations.crs,
        )
        return self

    def solve(self, stop=pyvrp.stop.NoImprovement(1e6), *args, **kwargs):
        """
        Solve a LastMile() instance according to the existing specification. 

        Parameters
        ----------
        stop    :   pyvrp.stop.StoppingCriterion
            A stopping rule that governs when the simulation will be ended. 
            Set to terminate solving after one million iterations with no improvement.


        Returns
        -------
        This LastMile() object, having added the results object to self.result_, as well
        as the routes and stops found to routes_ and stops_, respectively

        Notes
        -----
        other arguments and keyword arguments are passed directly to the pyvrp.Model.solve() method
        """
        if (not hasattr(self, "clients_")) | (not hasattr(self, "trucks_")):
            raise SpecificationError(
                "must assign both clients and trucks to" " solve a problem instance."
            )
        all_lonlats = numpy.row_stack(
            (self.depot_location, shapely.get_coordinates(self.clients_.geometry))
        )
        self._setup_graph(all_lonlats=all_lonlats)
        self.result_ = self.model.solve(stop=stop, *args, **kwargs)
        self.routes_, self.stops_ = utils.routes_and_stops(
            self.result_.best, self.model, self.clients_, self.depot_location, cost_unit=self.cost_unit
        )
        return self

    solve.__doc__ = pyvrp.Model.solve.__doc__

    def explore(self):
        """
        Make a webmap of the solution, colored by the route name. 
        """
        if not hasattr(self, "routes_"):
            raise SpecificationError("must have solved the model to show the result")
        m = self.routes_.sort_values("route_name").explore(
            "route_name", categorical=True, tiles="CartoDB positron"
        )
        stops_for_map = self.stops_.copy()
        stops_for_map["eta"] = self.stops_.eta.astype(str)
        stops_for_map.explore(
            "route_name",
            m=m,
            legend=False,
            style_kwds=dict(color="black", radius=3, weight=1.5),
        )
        geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(
                x=[self.depot_location[0]], y=[self.depot_location[1]], crs="epsg:4326"
            )
        ).explore(m=m, color="black", marker_type="marker")
        return m

    def _setup_graph(self, all_lonlats):
        """
        This sets up the graph pertaining to an inputted set of longitude and latitude coordinates. 

        Note that this assumes that there is a single vehicle profile.

        TODO: For multiple vehicle profiles, we would need to identify 
        the restricted and the base profiles, then update the model
        with an edge for each profile. 
        """
        raw_distances, raw_durations = engine.build_route_table(
            all_lonlats, all_lonlats, cost="both"
        )
        # how many minutes does it take to get from place to place?
        durations_by_block = numpy.ceil(raw_durations / 60)
        ##### WARNING!!!!!!! THIS IS A BUG IN OSRM #5855
        durations = numpy.clip(durations_by_block, 0, durations_by_block.max())
        distances = numpy.clip(raw_distances, 0, raw_distances.max()).round(0)

        duration_df = pandas.DataFrame(
            durations,
            index=[self.depot_name] + self.clients_.index.tolist(),
            columns=[self.depot_name] + self.clients_.index.tolist(),
        )
        distance_df = pandas.DataFrame(
            distances,
            index=[self.depot_name] + self.clients_.index.tolist(),
            columns=[self.depot_name] + self.clients_.index.tolist(),
        )
        for source_ix, source in enumerate(self.model.locations):
            for sink_ix, sink in enumerate(self.model.locations):
                self.model.add_edge(
                    source,
                    sink,
                    distance=distance_df.loc[source.name, sink.name].item(),
                    duration=duration_df.loc[
                        source.name, sink.name
                    ].item(),  # TODO: nogo zones
                )

    def write_result(
            self, filestem=None, write_geometries=False
        ):
        """
        Write the result of a solution out to three files: 
            1. the routes_ table is written to a file describing the route efficiency
            2. the stops_ table is written to a file describing each route
            3. the folium map from the .explore() method is written to a file

        This method requires that the model is solved first. 

        Parameters
        ----------
        filestem    :   str
            start of the name for all output files. If not provided, then the
            depot name is used as the default.
        write_geometries    : bool
            whether or not to write the geometries out to a file. If True, the
            output formats are geopackages. If False, as is default, then the
            output formats are csvs. The folium map is always written to an html file. 

        Returns
        -------
        the operation writes to disk and returns None.
        """
        if not hasattr(self, "routes_"):
            raise SpecificationError("Model must be solved before results can be written.")
        if write_geometries:
            def writer(df, filename):
                df.to_file(filename+".gpkg")
        else:
            def writer(df, filename):
                df.drop("geometry", axis=1, errors='ignore').to_csv(filename + ".csv")
        if filestem is None:
            filestem = self.depot_name.replace(" ", "_")
        writer(self.routes_, filestem+"_routes")
        writer(self.stops_, filestem+"_stops")
        self.explore().save(filestem+"_map.html")
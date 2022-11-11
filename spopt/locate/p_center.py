import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import BaseOutputMixin, LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist

import warnings


class PCenter(LocateSolver, BaseOutputMixin):
    """
    Implements and solve the optimal :math:`p`-center facility location problem.

    Parameters
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    aij : np.array
        A cost matrix in the form of a 2D array between origins and destinations.

    Attributes
    ----------

    name : str
        Problem name
    problem : pulp.LpProblem
        Pulp instance of optimization model that contains constraints,
        variables and objective function.
    fac2cli : np.array
        2-d array MxN, where m is number of facilities and n is number of clients.
        Each row represents a facility and has an array containing clients index
        meaning that the facility-i cover the entire array.
    cli2fac : np.array
        2-d MxN, where m is number of clients and n is number of facilities.
        Each row represent a client and has an array containing facility index
        meaning that the client is covered by the facility ith.
    aij : np.array
        A cost matrix in the form of a 2D array between origins and destinations.

    """

    def __init__(self, name: str, problem: pulp.LpProblem, aij: np.array):
        self.problem = problem
        self.name = name
        self.aij = aij

    def __add_obj(self) -> None:
        """
        Add objective function to the model:

        Minimize W

        Returns
        -------

        None

        """
        weight = getattr(self, "weight_var")

        self.problem += weight, "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        p_facilities: int,
        predefined_facilities_arr: np.array = None,
        name: str = "p-center",
    ):
        """
        Create a ``PCenter`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix: np.array
            A cost matrix in the form of a 2D array between origins and destinations.
        p_facilities: int
            number of facilities to be located
        predefined_facilities_arr : numpy.array
            Predefined facilities that must appear in the solution.
            Default is ``None``.
        name: str, default="p-center"
            name of the problem

        Returns
        -------

        spopt.locate.PCenter

        Examples
        --------

        >>> from spopt.locate import PCenter
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations.

        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["clients"],
        ...    destpattern=ntw.pointpatterns["facilities"]
        ... )

        Create and solve a ``PCenter`` instance from the cost matrix.

        >>> pcenter_from_cost_matrix = PCenter.from_cost_matrix(
        ...     cost_matrix, p_facilities=4
        ... )
        >>> pcenter_from_cost_matrix = pcenter_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility-client associations.

        >>> for fac, cli in enumerate(pcenter_from_cost_matrix.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 15 clients
        facility 1 serving 24 clients
        facility 2 serving 33 clients
        facility 3 serving 0 clients
        facility 4 serving 28 clients

        """
        r_cli = range(cost_matrix.shape[0])
        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMinimize)

        p_center = PCenter(name, model, cost_matrix)

        FacilityModelBuilder.add_facility_integer_variable(p_center, r_fac, "y[{i}]")
        FacilityModelBuilder.add_client_assign_integer_variable(
            p_center, r_cli, r_fac, "z[{i}_{j}]"
        )
        FacilityModelBuilder.add_weight_continuous_variable(p_center)

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                p_center, p_center.problem, predefined_facilities_arr
            )

        p_center.__add_obj()

        FacilityModelBuilder.add_facility_constraint(
            p_center, p_center.problem, p_facilities
        )
        FacilityModelBuilder.add_assignment_constraint(
            p_center, p_center.problem, r_fac, r_cli
        )
        FacilityModelBuilder.add_opening_constraint(
            p_center, p_center.problem, r_fac, r_cli
        )
        FacilityModelBuilder.add_minimized_maximum_constraint(
            p_center, p_center.problem, cost_matrix, r_fac, r_cli
        )

        return p_center

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        p_facilities: int,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "p-center",
    ):
        """
        Create a PCenter object based on geodataframes. Calculate the cost
        matrix between demand and facility, and then use from_cost_matrix method.

        Parameters
        ----------

        gdf_demand: geopandas.GeoDataFrame
            demand geodataframe with point geometry
        gdf_fac: geopandas.GeoDataframe
            facility geodataframe with point geometry
        demand_col: str
            demand geometry column name
        facility_col: str
            facility candidate sites geometry column name
        p_facilities: int
            number of facilities to be located
        predefined_facility_col: str
            Column name representing facilities are already defined.
            Default is ``None``.
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist`
            used for the distance calculations
        name: str, default="p-median"
            name of the problem

        Returns
        -------

        spopt.locate.PCenter

        Examples
        --------

        >>> from spopt.locate import PCenter
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as ``GeoDataFrame`` objects.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Create and solve a ``PCenter`` instance from the ``GeoDataFrame`` objects.

        >>> pcenter_from_geodataframe = PCenter.from_geodataframe(
        ...     clients_snapped,
        ...     facilities_snapped,
        ...     "geometry",
        ...     "geometry",
        ...     p_facilities=4,
        ...     distance_metric="euclidean"
        ... )
        >>> pcenter_from_geodataframe = pcenter_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility-client associations.

        >>> for fac, cli in enumerate(pcenter_from_geodataframe.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 14 clients
        facility 1 serving 26 clients
        facility 2 serving 34 clients
        facility 3 serving 0 clients
        facility 4 serving 26 clients

        """

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        _msg = (
            " geodataframe contains mixed type geometries or is not a point. Be "
            "sure deriving centroid from geometries doesn't affect the results."
        )
        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(f"Demand{_msg}", UserWarning)
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(f"Facility{_msg}", UserWarning)
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                "Geodataframes crs are different: "
                f"gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances, p_facilities, predefined_facilities_arr, name
        )

    def facility_client_array(self) -> None:
        """

        Create a 2D :math:`m \times n` array, where :math:`m` is number of
        facilities and :math:`n` is number of clients. Each row represent a
        facility and has an array containing a clients indices.

        Returns
        -------

        None

        """
        fac_vars = getattr(self, "fac_vars")
        cli_vars = getattr(self, "cli_assgn_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(len(cli_vars)):
                    if cli_vars[i][j].value() > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``PCenter`` model.

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.PCenter

        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()

        return self

import numpy as np

import pulp
from geopandas import GeoDataFrame

from .base import BaseOutputMixin, LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist

import warnings


class PCenter(LocateSolver, BaseOutputMixin):
    r"""
    Implement the :math:`p`-center optimization model and solve it. The
    :math:`p`-center problem, as adapted from :cite:`daskin_2013`,
    can be formulated as:

    .. math::

       \begin{array}{lllll}
       \displaystyle \textbf{Minimize}      & \displaystyle W                                   &&                                          & (1)                                                                               \\
       \displaystyle \textbf{Subject To}    & \displaystyle \sum_{j \in J}{X_{ij} = 1}          && \forall i \in I                          & (2)                                                                               \\
                                            & \displaystyle \sum_{j \in J}{Y_j} = p             &&                                          & (3)                                                                               \\
                                            & X_{ij} \leq Y_{j}                                 && \forall i \in I \quad \forall j \in J    & (4)                                                                               \\
                                            & W \geq \displaystyle \sum_{j \in J}{d_{ij}X_{ij}} && \forall i \in I                          & (5)                                                                               \\
                                            & X_{ij} \in \{0, 1\}                               && \forall i \in I \quad \forall j \in J    & (6)                                                                               \\
                                            & Y_j \in \{0, 1\}                                  && \forall j \in J                          & (7)                                                                               \\
                                            &                                                   &&                                          &                                                                                   \\
       \displaystyle \textbf{Where}         && i                                                & =                                         & \textrm{index of demand points/areas/objects in set } I                           \\
                                            && j                                                & =                                         & \textrm{index of potential facility sites in set } J                              \\
                                            && p                                                & =                                         & \textrm{the number of facilities to be sited}                                     \\
                                            && d_{ij}                                           & =                                         & \textrm{shortest distance or travel time between locations } i \textrm{ and } j   \\
                                            && X_{ij}                                           & =                                         & \begin{cases}
                                                                                                                                               1, \textrm{if client location } i \textrm{ is served by facility } j             \\
                                                                                                                                               0, \textrm{otherwise}                                                            \\
                                                                                                                                              \end{cases}                                                                       \\
                                            && Y_j                                              & =                                         & \begin{cases}
                                                                                                                                               1, \textrm{if a facility is sited at location } j                                \\
                                                                                                                                               0, \textrm{otherwise}                                                            \\
                                                                                                                                              \end{cases}                                                                      \\
                                            && W                                                & =                                         & \textrm{maximum distance between any demand site and its associated facility}
       \end{array}

    Parameters
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    aij : numpy.array
        A cost matrix in the form of a 2D array between origins and destinations.

    Attributes
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    fac2cli : numpy.array
        A 2D array storing facility to client relationships where each
        row represents a facility and contains an array of client indices
        with which it is associated. An empty client array indicates
        the facility is associated with no clients.
    cli2fac : numpy.array
        The inverse of ``fac2cli`` where client to facility relationships
        are shown.
    aij : numpy.array
        A cost matrix in the form of a 2D array between origins and destinations.

    """  # noqa

    def __init__(self, name: str, problem: pulp.LpProblem, aij: np.array):
        self.problem = problem
        self.name = name
        self.aij = aij

    def __add_obj(self) -> None:
        """
        Add the objective function to the model.

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

        cost_matrix: numpy.array
            A cost matrix in the form of a 2D array between origins and destinations.
        p_facilities : int
            The number of facilities to be located.
        predefined_facilities_arr : numpy.array (default None)
            Predefined facilities that must appear in the solution.
        name : str (default 'p-center')
            The problem name.

        Returns
        -------

        spopt.locate.p_center.PCenter

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
        FacilityModelBuilder.add_client_assign_variable(
            p_center, r_cli, r_fac, "z[{i}_{j}]"
        )
        FacilityModelBuilder.add_weight_continuous_variable(p_center)

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                p_center, predefined_facilities_arr
            )

        p_center.__add_obj()

        FacilityModelBuilder.add_facility_constraint(p_center, p_facilities)
        FacilityModelBuilder.add_assignment_constraint(p_center, r_fac, r_cli)
        FacilityModelBuilder.add_opening_constraint(p_center, r_fac, r_cli)
        FacilityModelBuilder.add_minimized_maximum_constraint(
            p_center, cost_matrix, r_fac, r_cli
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

        Create an ``PCenter`` object from ``geopandas.GeoDataFrame`` objects.
        Calculate the cost matrix between demand and facility locations
        before building the problem within the ``from_cost_matrix()`` method.

        Parameters
        ----------

        gdf_demand : geopandas.GeoDataFrame
            Demand locations.
        gdf_fac : geopandas.GeoDataFrame
            Facility locations.
        demand_col : str
            Demand sites geometry column name.
        facility_col : str
            Facility candidate sites geometry column name.
        p_facilities: int
           The number of facilities to be located.
        predefined_facility_col : str (default None)
            Column name representing facilities are already defined.
        distance_metric : str (default 'euclidean')
            A metric used for the distance calculations supported by
            `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_.
        name : str (default 'p-center')
            The name of the problem.

        Returns
        -------

        spopt.locate.p_center.PCenter

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

        """  # noqa

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

        Create a 2D array storing **facility to client relationships** where each
        row represents a facility and contains an array of client indices
        with which it is associated. An empty client array indicates
        the facility is associated with no clients.

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
                    if cli_vars[i, j].value() > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``PCenter`` model.

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool (default True)
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.p_center.PCenter

        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()

        return self

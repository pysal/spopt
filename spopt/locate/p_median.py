# ruff: noqa: B009, B010

import warnings

import numpy as np
import pulp
from geopandas import GeoDataFrame
from pointpats.geometry import build_best_tree
from scipy.sparse import csr_matrix, find
from scipy.spatial.distance import cdist

from .base import (
    BaseOutputMixin,
    FacilityModelBuilder,
    LocateSolver,
    MeanDistanceMixin,
    SpecificationError,
)


class PMedian(LocateSolver, BaseOutputMixin, MeanDistanceMixin):
    r"""
    Implement the :math:`p`-median optimization model and solve it. The
    :math:`p`-median problem, as adapted from :cite:`daskin_2013`,
    can be formulated as:

    .. math::

       \begin{array}{lllll}
       \displaystyle \textbf{Minimize}      & \displaystyle \sum_{i \in I}\sum_{j \in J}{a_i d_{ij} X_{ij}} &&                                          & (1)                                                                               \\
       \displaystyle \textbf{Subject To}    & \displaystyle \sum_{j \in J}{X_{ij} = 1}                      && \forall i \in I                          & (2)                                                                               \\
                                            & \displaystyle \sum_{j \in J}{Y_j} = p                         &&                                          & (3)                                                                               \\
                                            & X_{ij} \leq Y_{j}                                             && \forall i \in I \quad \forall j \in J    & (4)                                                                               \\
                                            & X_{ij} \in \{0, 1\}                                           && \forall i \in I \quad \forall j \in J    & (5)                                                                               \\
                                            & Y_j \in \{0, 1\}                                              && \forall j \in J                          & (6)                                                                               \\
                                            &                                                               &&                                          &                                                                                   \\
       \displaystyle \textbf{Where}         && i                                                            & =                                         & \textrm{index of demand points/areas/objects in set } I                           \\
                                            && j                                                            & =                                         & \textrm{index of potential facility sites in set } J                              \\
                                            && p                                                            & =                                         & \textrm{the number of facilities to be sited}                                     \\
                                            && a_i                                                          & =                                         & \textrm{service load or population demand at client location } i \\
                                            && d_{ij}                                                       & =                                         & \textrm{shortest distance or travel time between locations } i \textrm{ and } j   \\
                                            && X_{ij}                                                       & =                                         & \begin{cases}
                                                                                                                                                           1, \textrm{if client location } i \textrm{ is served by facility } j             \\
                                                                                                                                                           0, \textrm{otherwise}                                                            \\
                                                                                                                                                          \end{cases}                                                                       \\
                                            && Y_j                                                          & =                                         & \begin{cases}
                                                                                                                                                           1, \textrm{if a facility is sited at location } j                                \\
                                                                                                                                                           0, \textrm{otherwise}                                                            \\
                                                                                                                                                          \end{cases}                                                                       \\
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

    """  # noqa: E501

    def __init__(
        self,
        name: str,
        problem: pulp.LpProblem,
        aij: np.array,
        weights_sum: int | float,
    ):
        self.aij = aij
        self.ai_sum = weights_sum
        self.name = name
        self.problem = problem

    def __add_obj(self, range_clients: range, range_facility: range) -> None:
        """
        Add the objective function to the model.

        Minimize s0_0 * z0_0 + s0_1 * z0_1 + ... + si_j * zi_j

        Parameters
        ----------

        range_clients : range
            The range of demand points.
        range_facility : range
            The range of facility point.

        Returns
        -------

        None

        """
        cli_assgn_vars = getattr(self, "cli_assgn_vars")

        self.problem += (
            pulp.lpSum(
                [
                    self.aij[i, j] * cli_assgn_vars[i, j]
                    for i in range_clients
                    for j in range_facility
                ]
            ),
            "objective function",
        )

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        weights: np.array,
        p_facilities: int,
        predefined_facilities_arr: np.array = None,
        facility_capacities: np.array = None,
        fulfill_predefined_fac: bool = False,
        name: str = "p-median",
    ):
        """
        Create a ``PMedian`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix : numpy.array
            A cost matrix in the form of a 2D array between origins and destinations.
        weights : numpy.array
            A 1D array of service load or population demand.
        p_facilities : int
            The number of facilities to be located.
        predefined_facilities_arr : numpy.array (default None)
            A binary 1D array of service facilities that must appear in the
            solution. For example, consider 3 facilites ``['A', 'B', 'C']``.
            If facility ``'B'`` must be in the model solution, then the passed
            in array should be ``[0, 1, 0]``.
        facility_capacity : numpy.array (default None)
            The capacity of each facility.
        fulfill_predefined_fac : bool (default False)
            If the predefined facilities need to be fulfilled.
        name : str (default 'p-median')
            The problem name.

        Returns
        -------

        spopt.locate.p_median.PMedian

        Examples
        --------

        >>> from spopt.locate import PMedian
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
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

        Simulate demand weights from ``1`` to ``12``.

        >>> ai = numpy.random.randint(1, 12, 100)

        Create and solve a ``PMedian`` instance from the cost matrix.

        >>> pmedian_from_cost_matrix = PMedian.from_cost_matrix(
        ...     cost_matrix, ai, p_facilities=4
        ... )
        >>> pmedian_from_cost_matrix = pmedian_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility-client associations.

        >>> for fac, cli in enumerate(pmedian_from_cost_matrix.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 14 clients
        facility 1 serving 29 clients
        facility 2 serving 31 clients
        facility 3 serving 0 clients
        facility 4 serving 26 clients

        Get the total and average weighted travel cost.

        >>> round(pmedian_from_cost_matrix.problem.objective.value(), 3)
        1870.747
        >>> round(pmedian_from_cost_matrix.mean_dist, 3)
        3.027

        """
        n_cli = cost_matrix.shape[0]
        r_cli = range(n_cli)
        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMinimize)

        weights_sum = weights.sum()
        weights = np.reshape(weights, (n_cli, 1))
        aij = weights * cost_matrix

        p_median = PMedian(name, model, aij, weights_sum)

        FacilityModelBuilder.add_facility_integer_variable(p_median, r_fac, "y[{i}]")
        FacilityModelBuilder.add_client_assign_variable(
            p_median, r_cli, r_fac, "z[{i}_{j}]"
        )

        if predefined_facilities_arr is not None:
            if fulfill_predefined_fac and facility_capacities is not None:
                sum_predefined_fac_cap = np.sum(
                    facility_capacities[predefined_facilities_arr]
                )
                if sum_predefined_fac_cap <= weights.sum():
                    FacilityModelBuilder.add_predefined_facility_constraint(
                        p_median,
                        predefined_facilities_arr,
                        weights,
                        facility_capacities,
                    )
                else:
                    raise SpecificationError(
                        "Problem is infeasible. The predefined facilities can't be "
                        "fulfilled, because their capacity is larger than the total "
                        f"demand {weights.sum()}."
                    )
            elif fulfill_predefined_fac and facility_capacities is None:
                raise SpecificationError(
                    "Data on the capacity of the facility is missing, "
                    "so the model cannot be calculated."
                )
            else:
                FacilityModelBuilder.add_predefined_facility_constraint(
                    p_median, predefined_facilities_arr
                )

        if facility_capacities is not None:
            sorted_capacities = np.sort(facility_capacities)
            highest_possible_capacity = sorted_capacities[-p_facilities:].sum()
            if highest_possible_capacity < weights.sum():
                raise SpecificationError(
                    "Problem is infeasible. The highest possible capacity "
                    f"{highest_possible_capacity}, coming from the {p_facilities} "
                    "sites with the highest capacity, is smaller than "
                    f"the total demand {weights.sum()}."
                )
            FacilityModelBuilder.add_facility_capacity_constraint(
                p_median,
                weights,
                facility_capacities,
                range(len(weights)),
                range(len(facility_capacities)),
            )
        p_median.__add_obj(r_cli, r_fac)

        FacilityModelBuilder.add_facility_constraint(p_median, p_facilities)
        FacilityModelBuilder.add_assignment_constraint(p_median, r_fac, r_cli)
        FacilityModelBuilder.add_opening_constraint(p_median, r_fac, r_cli)

        return p_median

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        weights_cols: str,
        p_facilities: int,
        facility_capacity_col: str = None,
        predefined_facility_col: str = None,
        fulfill_predefined_fac: bool = False,
        distance_metric: str = "euclidean",
        name: str = "p-median",
    ):
        """

        Create an ``PMedian`` object from ``geopandas.GeoDataFrame`` objects.
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
        weights_cols : str
            The weight column name representing service load or demand.
        p_facilities: int
            The number of facilities to be located.
        predefined_facility_col : str (default None)
            Column name representing facilities are already defined.
            This a binary assignment per facility. For example, consider 3 facilites
            ``['A', 'B', 'C']``. If facility ``'B'`` must be in the model solution,
            then the column should be ``[0, 1, 0]``.
        facility_capacities_col: str (default None)
            Column name representing the capacities of each facility.
        fulfill_predefined_fac : bool (default False)
            If the predefined facilities need to be fulfilled.
        distance_metric : str (default 'euclidean')
            A metric used for the distance calculations supported by
            `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_.
        name : str (default 'p-median')
            The name of the problem.

        Returns
        -------

        spopt.locate.p_median.PMedian

        Examples
        --------

        >>> from spopt.locate import PMedian
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
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

        Simulate demand weights from ``1`` to ``12``.

        >>> ai = numpy.random.randint(1, 12, 100)
        >>> clients_snapped['weights'] = ai

        Create and solve a ``PMedian`` instance from the ``GeoDataFrame`` object.

        >>> pmedian_from_geodataframe = PMedian.from_geodataframe(
        ...    clients_snapped,
        ...    facilities_snapped,
        ...    "geometry",
        ...    "geometry",
        ...    "weights",
        ...    p_facilities=4,
        ...    distance_metric="euclidean"
        ... )
        >>> pmedian_from_geodataframe = pmedian_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility-client associations.

        >>> for fac, cli in enumerate(pmedian_from_geodataframe.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 13 clients
        facility 1 serving 29 clients
        facility 2 serving 31 clients
        facility 3 serving 0 clients
        facility 4 serving 27 clients

        """  # noqa: E501

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        facility_capacities = None
        if facility_capacity_col is not None:
            facility_capacities = gdf_fac[facility_capacity_col].to_numpy()

        service_load = gdf_demand[weights_cols].to_numpy()
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        _msg = (
            " geodataframe contains mixed type geometries or is not a point. Be "
            "sure deriving centroid from geometries doesn't affect the results."
        )
        if len(dem_type_geom) > 1 or "Point" not in dem_type_geom:
            warnings.warn(f"Demand{_msg}", UserWarning, stacklevel=2)
            dem = dem.centroid

        if len(fac_type_geom) > 1 or "Point" not in fac_type_geom:
            warnings.warn(f"Facility{_msg}", UserWarning, stacklevel=2)
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
            cost_matrix=distances,
            weights=service_load,
            p_facilities=p_facilities,
            predefined_facilities_arr=predefined_facilities_arr,
            facility_capacities=facility_capacities,
            fulfill_predefined_fac=fulfill_predefined_fac,
            name=("capacitated-" + name if facility_capacities is not None else name),
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
        Solve the ``PMedian`` model.

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool (default True)
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.p_median.PMedian

        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()
            self.get_mean_distance()

        return self


class KNearestPMedian(PMedian):
    r"""
    Implement the P-Median Model with Near-Far Cost Allocation and solve it.
    The model is adapted from :cite:`church_2018`, and can be formulated as:

    .. math::

       \begin{array}{lllll}
       \displaystyle \textbf{Minimize}      & \displaystyle \sum_{i \in I}\sum_{k \in k_{i}}{a_i d_{ik} X_{ik}} + \sum_{i \in I}{g_i (d_{i{k_i}} + 1)}  &&                                          & (1)                                                                               \\
       \displaystyle \textbf{Subject To}    & \sum_{k \in k_{i}}{X_{ik} + g_i = 1}                                                                      && \forall i \in I                          & (2)                                                                               \\
                                            & \sum_{j \in J}{Y_j} = p                                                                                   &&                                          & (3)                                                                               \\
                                            & \sum_{i \in I}{a_i X_{ik}} \leq {Y_{k} c_{k}}                                                             && \forall k \in k_{i}                      & (4)                                                                               \\
                                            & X_{ij} \leq Y_{j}                                                                                         && \forall i \in I \quad \forall j \in J    & (5)                                                                               \\
                                            & X_{ij} \in \{0, 1\}                                                                                       && \forall i \in I \quad \forall j \in J    & (6)                                                                               \\
                                            & Y_j \in \{0, 1\}                                                                                          && \forall j \in J                          & (7)                                                                               \\
                                            &                                                                                                           &&                                          &                                                                                   \\
       \displaystyle \textbf{Where}         && i                                                                                                        & =                                         & \textrm{index of demand points/areas/objects in set } I                           \\
                                            && j                                                                                                        & =                                         & \textrm{index of potential facility sites in set } J                              \\
                                            && p                                                                                                        & =                                         & \textrm{the number of facilities to be sited}                                     \\
                                            && a_i                                                                                                      & =                                         & \textrm{service load or population demand at client location } i                  \\
                                            && k_{i}                                                                                                    & =                                         & \textrm{the } k \textrm{nearest facilities of client location } i                 \\
                                            && c_{j}                                                                                                    & =                                         & \textrm{the capacity of facility} j                                               \\
                                            && d_{ij}                                                                                                   & =                                         & \textrm{shortest distance or travel time between locations } i \textrm{ and } j   \\
                                            && X_{ij}                                                                                                   & =                                         & \begin{cases}
                                                                                                                                                                                                       1, \textrm{if client location } i \textrm{ is served by facility } j             \\
                                                                                                                                                                                                       0, \textrm{otherwise}                                                            \\
                                                                                                                                                                                                      \end{cases}                                                                       \\
                                            && Y_j                                                                                                      & =                                         & \begin{cases}
                                                                                                                                                                                                       1, \textrm{if a facility is sited at location } j                                \\
                                                                                                                                                                                                       0, \textrm{otherwise}                                                            \\
                                                                                                                                                                                                      \end{cases}                                                                       \\
                                            && g_i                                                                                                      & =                                         & \begin{cases}
                                                                                                                                                                                                       1, \textrm{if the client } i \textrm{ needs to be served by non-k-nearest facilities}     \\
                                                                                                                                                                                                       0, \textrm{otherwise}                                                            \\
                                                                                                                                                                                                      \end{cases}                                                                       \\
       \end{array}

    Parameters
    ----------

    name : str
        The problem name.
    ai_sum : Union[int, float]
        The sum of weights representing the service loads of the clients.
    clients : np.array
        An array of coordinates of clients.
    facilities : np.array
        An array of coordinates of facilities.
    weights : np.array
        An array of weights representing the service loads of the clients.
    k_array : np.array
        An array of k values representing the number of nearest facilities
        for each client.
    p_facilities: int
        The number of facilities to be located.
    capacities : np.array or None
        An array of facility capacities. None if capacity constraints are
        not considered.
    distance_metric : str
        The distance metric used for computing distances between clients
        and facilities.

    Attributes
    ----------

    sparse_matrix : Compressed Sparse Row matrix
        A cost matrix in the form of a compressed sparse row matrix between
        origins and destinations.
    aij : Compressed Sparse Row matrix
        A weighted cost matrix in the form of a compressed sparse row matrix
        between origins and destinations.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains constraints,
        variables, and an objective function.
    fac2cli : numpy.array
        A 2D array storing facility to client relationships where each
        row represents a facility and contains an array of client indices
        with which it is associated. An empty client array indicates
        the facility is associated with no clients.
    cli2fac : numpy.array
        The inverse of ``fac2cli`` where client to facility relationships
        are shown.

    """  # noqa: E501

    def __init__(
        self,
        name: str,
        ai_sum: int | float,
        clients: np.array,
        facilities: np.array,
        weights: np.array,
        k_array: np.array,
        p_facilities: int,
        capacities: np.array = None,
        distance_metric: str = "euclidean",
    ):
        self.ai_sum = ai_sum
        self.clients = clients
        self.facilities = facilities
        self.weights = weights
        self.k_array = k_array
        self.p_facilities = p_facilities
        self.capacities = capacities
        self.distance_metric = distance_metric
        self.name = name

    def __add_obj(
        self, max_distance: np.array, range_clients: range, range_facility: range
    ) -> None:
        """
        Add the objective function to the model.

        Parameters
        ----------

        max_distance : np.array
            An array of distances between each client and their k nearest facility.
            For example, when k = 2, this array will only store the distances between
            each client and their nearest and second nearest facility.
        range_clients : range
            The range of demand points.
        range_facility : range
            The range of facility point.

        Returns
        -------

        None

        """
        cli_assgn_vars = getattr(self, "cli_assgn_vars")
        placeholder_vars = getattr(self, "placeholder_vars")

        self.problem += (
            pulp.lpSum(
                pulp.lpSum(
                    self.aij[i, j] * cli_assgn_vars.get((i, j), 0)
                    for j in range_facility
                )
                + (placeholder_vars[i] * (max_distance[i] + 1))
                for i in range_clients
            ),
            "objective function",
        )

    @classmethod
    def from_cost_matrix(cls, *args, **kwargs):
        """
        Warning: This method is not supported in the KNearestPMedian subclass.
        """
        raise NotImplementedError(
            "The `from_cost_matrix()` method is not "
            "supported in `KNearestPMedian` class."
        )

    def _create_sparse_matrix(self) -> None:
        """
        Create a sparse matrix representing the distance between clients
        and their k nearest facilities.

        This method uses a suitable tree data structure (built with the
        ``pointpats.geometry.build_best_tree()`` function) to efficiently
        find the ``k`` nearest facilities for each client based on the
        specified distance metric. The resulting distances are stored in a
        sparse matrix format to conserve memory for large datasets.

        Returns
        -------

        None
        """

        row_shape = len(self.clients)
        column_shape = len(self.facilities)

        # check the k value with the total number of facilities
        if not (self.k_array <= column_shape).all():
            raise ValueError(
                "The value of `k` should be no more than the number "
                f"of total facilities: ({column_shape})."
            )

        # Initialize empty lists to store the data for the sparse matrix
        data = []
        row_index = []
        col_index = []

        # create the suitable Tree
        tree = build_best_tree(self.facilities, self.distance_metric)

        for i, k in enumerate(self.k_array):
            # Query the Tree to find the k nearest facilities for each client
            distances, k_nearest_facilities_indices = tree.query([self.clients[i]], k=k)

            # extract the contents of the inner array
            distances = distances[0].tolist()
            k_nearest_facilities_indices = k_nearest_facilities_indices[0].tolist()

            # Append the data for the sparse matrix
            data.extend(distances)
            row_index.extend([i] * k)
            col_index.extend(k_nearest_facilities_indices)

        # Create the sparse matrix using csr_matrix
        self.sparse_matrix = csr_matrix(
            (data, (row_index, col_index)), shape=(row_shape, column_shape)
        )

    def _update_k_array(self) -> None:
        """
        Increase the ``k`` value for clients with any :math:`g_i > 0`
        and update the ``k`` array.

        This method is used to adjust the ``k`` values for clients based on their
        placeholder variable :math:`g_i`. For clients with :math:`g_i` greater
        than ``0``, the corresponding ``k`` value is increased by ``1`` in the
        new ``k`` array.

        Returns
        -------

        None
        """

        new_k_array = self.k_array.copy()
        placeholder_vars = getattr(self, "placeholder_vars")
        for i in range(len(placeholder_vars)):
            if placeholder_vars[i].value() > 0:
                new_k_array[i] = new_k_array[i] + 1
        self.k_array = new_k_array

    def _from_sparse_matrix(self) -> None:
        """
        Create the k nearest p-median problem from the sparse distance matrix.

        Returns
        -------

        None
        """
        n_cli = self.sparse_matrix.shape[0]
        r_cli = range(n_cli)
        r_fac = range(self.sparse_matrix.shape[1])

        self.weights = np.reshape(self.weights, (n_cli, 1))
        # create and store the weighted sparse matrix
        self.aij = self.sparse_matrix.multiply(self.weights).tocsr()

        # create the model/problem
        self.problem = pulp.LpProblem(self.name, pulp.LpMinimize)

        # add all the 1)decision variable, 2)objective function, and 3)constraints

        # Facility integer decision variable
        FacilityModelBuilder.add_facility_integer_variable(self, r_fac, "y[{i}]")
        fac_vars = getattr(self, "fac_vars")

        # Placeholder facility decision variable
        placeholder_vars = pulp.LpVariable.dicts(
            "g", (i for i in r_cli), 0, 1, pulp.LpBinary
        )
        setattr(self, "placeholder_vars", placeholder_vars)

        # Client assignment integer decision variables
        row_indices, col_indices, values = find(self.aij)

        cli_assgn_vars = pulp.LpVariable.dicts(
            "z", list(zip(row_indices, col_indices, strict=True)), 0, 1, pulp.LpBinary
        )
        setattr(self, "cli_assgn_vars", cli_assgn_vars)

        # Add the objective function
        max_distance = self.aij.max(axis=1).toarray().flatten()
        self.__add_obj(max_distance, r_cli, r_fac)

        # Create the capacity constraints
        if self.capacities is not None:
            sorted_capacities = np.sort(self.capacities)
            highest_possible_capacity = sorted_capacities[-self.p_facilities :].sum()
            if highest_possible_capacity < self.ai_sum:
                raise SpecificationError(
                    "Problem is infeasible. The highest possible capacity "
                    f"({highest_possible_capacity}), coming from the "
                    f"{self.p_facilities} sites with the highest capacity, "
                    f"is smaller than the total demand ({self.ai_sum})."
                )
            for j in col_indices:
                self.problem += (
                    pulp.lpSum(
                        self.weights[i] * cli_assgn_vars.get((i, j), 0) for i in r_cli
                    )
                    <= fac_vars[j] * self.capacities[j]
                )

        # Create assignment constraints.
        for i in r_cli:
            self.problem += (
                pulp.lpSum(cli_assgn_vars.get((i, j), 0) for j in set(col_indices))
                + placeholder_vars[i]
                == 1
            )
        # Create the facility constraint
        FacilityModelBuilder.add_facility_constraint(self, self.p_facilities)

        # Create opening constraints
        FacilityModelBuilder.add_opening_constraint(self, r_fac, r_cli)

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        weights_cols: str,
        p_facilities: int,
        facility_capacity_col: str = None,
        k_array: np.array = None,
        distance_metric: str = "euclidean",
        name: str = "k-nearest-p-median",
    ):
        """
        Create the object of KNearestPMedian class using input data.

        Parameters
        ----------
        gdf_demand : GeoDataFrame
            A GeoDataFrame containing demand points with their associated attributes.
        gdf_fac : GeoDataFrame
            A GeoDataFrame containing facility points with their associated attributes.
        demand_col : str
            The column name in gdf_demand representing the coordinate of each client.
        facility_col : str
            The column name in gdf_fac representing the coordinate of each facility.
        weights_cols : str
            The column name in gdf_demand representing the weights for each client.
        p_facilities: int
           The number of facilities to be located.
        facility_capacity_col : str, optional
            The column name in gdf_fac representing the capacity of each facility,
            by default None.
        k_array : np.array, optional
            An array of integers representing the k values for each client.
            If not provided, a default value of 5 or the number of facilities,
            whichever is smaller, will be used.
        distance_metric : str, optional
            The distance metric to be used in calculating distances between clients
            and facilities, by default "euclidean".
        name : str, optional
            The name of the problem, by default "k-nearest-p-median".

        Returns
        -------

        spopt.locate.p_median.KNearestPMedian

        Examples
        --------
        >>> from spopt.locate import KNearestPMedian
        >>> import geopandas

        Create the input data and attributes.

        >>> k = np.array([1, 1])
        >>> demand_data = {
        ...    'ID': [1, 2],
        ...    'geometry': [Point(0.5, 1), Point(1.5, 1)],
        ...    'demand': [1, 1]}
        >>> facility_data = {
        ...    'ID': [101, 102],
        ...    'geometry': [Point(1,1), Point(0, 2), Point(2, 0)],
        ...    'capacity': [1, 1, 1]}
        >>> gdf_demand = geopandas.GeoDataFrame(demand_data, crs='EPSG:4326')
        >>> gdf_fac = geopandas.GeoDataFrame(facility_data, crs='EPSG:4326')

        Create and solve a ``KNearestPMedian`` instance from the geodataframe.

        >>> k_nearest_pmedian = KNearestPMedian.from_geodataframe(
        ...     gdf_demand, gdf_fac,'geometry','geometry', weights_cols='demand',
        ...     2, facility_capacity_col='capacity', k_array = k)
        >>> k_nearest_pmedian = k_nearest_pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        Get the facility-client associations.

        >>> for fac, cli in enumerate(k_nearest_pmedian.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 1 clients
        facility 1 serving 1 clients
        facility 2 serving 0 clients

        Get the total and average weighted travel cost.

        >>> round(k_nearest_pmedian.problem.objective.value(), 3)
        1.618
        >>> round(k_nearest_pmedian.mean_dist, 3)
        0.809

        Get the k values for the last iteration.

        >>> print(k_nearest_pmedian.k_array)
        [2, 1]

        """

        # check the crs of two geodataframes
        if gdf_demand.crs is None:
            raise ValueError("GeoDataFrame ``gdf_demand`` does not have a valid CRS.")
        if gdf_fac.crs is None:
            raise ValueError("GeoDataFrame ``gdf_fac`` does not have a valid CRS.")
        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                "Geodataframes crs are different: "
                f"gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        # create the array of coordinate of clients and facilities
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]
        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        # check the values of k_array
        if k_array is None:
            k_array = np.full(len(dem_data), np.minimum(len(fac_data), 5))
        elif not isinstance(k_array, np.ndarray):
            raise TypeError("`k_array` should be a numpy array.")
        elif not (k_array <= len(fac_data)).all():
            raise ValueError(
                f"The value of `k` should be no more than the number "
                f"of total facilities, which is {len(fac_data)}."
            )

        # demand and capacity
        service_load = gdf_demand[weights_cols].to_numpy()
        weights_sum = service_load.sum()
        facility_capacities = None
        if facility_capacity_col is not None:
            facility_capacities = gdf_fac[facility_capacity_col].to_numpy()

        return KNearestPMedian(
            ("capacitated-" + name if facility_capacities is not None else name),
            weights_sum,
            dem_data,
            fac_data,
            service_load,
            k_array,
            p_facilities,
            facility_capacities,
            distance_metric,
        )

    def facility_client_array(self) -> None:
        """

        Create a 2D array storing **facility to client relationships** where each
        row represents a facility and contains an array of client indices with
        which it is associated. An empty client array indicates the facility
        is associated with no clients.

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
                    if (i, j) in cli_vars and cli_vars[i, j].value() > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """

        Solve the k-nearest p-median model.

        This method iteratively solves the ``KNearestPMedian`` model using a specified
        solver until no more clients need to be assigned to placeholder facilities.
        The ``k`` values for clients are increased dynamically based on the presence of
        clients not assigned to ``k`` nearest facilities.

        Parameters
        ----------
        solver : pulp.LpSolver
            The solver to be used for solving the optimization model.
        results : bool (default True)
            If ``True`` it will create metainfo (which facilities cover which
            demand) and vice versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.p_median.KNearestPMedian

        """

        # initialize the sum of the values of g_i
        sum_gi = 1

        # start the loop
        while sum_gi > 0:
            self._create_sparse_matrix()
            self._from_sparse_matrix()
            self.problem.solve(solver)
            self.check_status()

            # check the result
            placeholder_vars = getattr(self, "placeholder_vars")
            sum_gi = sum(
                placeholder_vars[i].value()
                for i in range(len(placeholder_vars))
                if placeholder_vars[i].value() > 0
            )
            if sum_gi > 0:
                self._update_k_array()

        if results:
            self.facility_client_array()
            self.client_facility_array()
            self.get_mean_distance()

        return self

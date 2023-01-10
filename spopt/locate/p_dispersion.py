import numpy as np

import pulp
from geopandas import GeoDataFrame

from .base import LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist

import warnings


class PDispersion(LocateSolver):
    r"""
    Implement the :math:`p`-dispersion optimization model and solve it
    :cite:`kuby_1987`. The :math:`p`-dispersion problem, as adapted from
    :cite:`MALISZEWSKI2012331`, can be formulated as:

    .. math::

       \begin{array}{lllll}
       \displaystyle \textbf{Maximize}      & \displaystyle D                       &&                                      & (1)                                                                               \\
       \displaystyle \textbf{Subject To}    & \displaystyle \sum_{i \in I}{Y_i} = p &&                                      & (2)                                                                               \\
                                            & D \leq d_{ij} + M (2 - Y_{i} - Y_{j}) && \forall i \in I \quad \forall j > i  & (3)                                                                               \\
                                            & Y_i \in \{0, 1\}                      && \forall i \in I                      & (4)                                                                               \\
                                            &                                       &&                                      &                                                                                   \\
       \displaystyle \textbf{Where}         && i, j                                 & =                                     & \textrm{index of potential facility sites in set } I                              \\
                                            && p                                    & =                                     & \textrm{the number of facilities to be sited}                                     \\
                                            && d_{ij}                               & =                                     & \textrm{shortest distance or travel time between locations } i \textrm{ and } j   \\
                                            && D                                    & =                                     & \textrm{minimum distance between any two sited facilities } i \textrm{ and } j    \\
                                            && M                                    & =                                     & \textrm{some large number; such that } M \geq \max_{ij}\{d_{ij}\}                 \\
                                            && Y_i                                  & =                                     & \begin{cases}
                                                                                                                               1, \textrm{if a facility is sited at location } i                                \\
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

    Attributes
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.

    """  # noqa

    def __init__(self, name: str, problem: pulp.LpProblem, p_facilities: int):
        self.p_facilities = p_facilities
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add the objective function to the model.

        Maximize D

        Returns
        -------

        None

        """
        disperse = getattr(self, "disperse_var")

        self.problem += disperse, "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        p_facilities: int,
        predefined_facilities_arr: np.array = None,
        name: str = "P-Dispersion",
    ):
        """
        Create a ``PDispersion`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix : np.array
            A cost matrix in the form of a 2D array between origins and destinations.
        p_facilities : int
            The number of facilities to be located.
        predefined_facilities_arr : numpy.array (default None)
            Predefined facilities that must appear in the solution.
        name : str (default 'P-Dispersion')
            The name of the problem.

        Returns
        -------

        spopt.locate.p_dispersion.PDispersion

        Examples
        --------

        >>> from spopt.locate.p_dispersion import PDispersion
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

        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations. Origins
        and destinations are both ``'facilities'`` in this case.

        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["facilities"],
        ...    destpattern=ntw.pointpatterns["facilities"]
        ... )

        Create and solve a ``PDispersion`` instance from the cost matrix.

        >>> pdispersion_from_cost_matrix = PDispersion.from_cost_matrix(
        ...     cost_matrix, p_fac=2
        ... )
        >>> pdispersion_from_cost_matrix = pdispersion_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Examine the solution.

        >>> for dv in pdispersion_from_cost_matrix.fac_vars:
        ...     if dv.varValue:
        ...         print(f"facility {dv.name} is selected")
        facility y_0_ is selected
        facility y_1_ is selected

        """

        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMaximize)
        p_dispersion = PDispersion(name, model, p_facilities)

        FacilityModelBuilder.add_maximized_min_variable(p_dispersion)
        p_dispersion.__add_obj()

        FacilityModelBuilder.add_facility_integer_variable(
            p_dispersion, r_fac, "y[{i}]"
        )

        FacilityModelBuilder.add_facility_constraint(
            p_dispersion, p_dispersion.p_facilities
        )

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                p_dispersion, predefined_facilities_arr
            )

        FacilityModelBuilder.add_p_dispersion_interfacility_constraint(
            p_dispersion, cost_matrix, r_fac
        )

        return p_dispersion

    @classmethod
    def from_geodataframe(
        cls,
        gdf_fac: GeoDataFrame,
        facility_col: str,
        p_facilities: int,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "P-Dispersion",
    ):
        """
        Create a ``PDispersion`` object based on a geodataframe. Calculate the
        cost matrix between facilities, and then use the from_cost_matrix method.

        Parameters
        ----------

        gdf_fac : geopandas.GeoDataFrame
            Facility locations.
        facility_col : str
            Facility candidate sites geometry column name.
        p_facilities : int
           The number of facilities to be located.
        predefined_facility_col : str (default None)
            Column name representing facilities are already defined.
        distance_metric : str (default 'euclidean')
            A metric used for the distance calculations supported by
            `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_.
        name : str (default 'P-Dispersion')
            The name of the problem.

        Returns
        -------

        spopt.locate.p_dispersion.PDispersion

        Examples
        --------

        >>> from spopt.locate.p_dispersion import PDispersion
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

        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as a ``GeoDataFrame`` object.

        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Create and solve a ``PDispersion`` instance from the ``GeoDataFrame`` object.

        >>> pdispersion_from_geodataframe = PDispersion.from_geodataframe(
        ...     facilities_snapped,
        ...     "geometry",
        ...     p_fac=2,
        ...     distance_metric="euclidean"
        ... )
        >>> pdispersion_from_geodataframe = pdispersion_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Examine the solution.

        >>> for dv in pdispersion_from_geodataframe.fac_vars:
        ...     if dv.varValue:
        ...         print(f"facility {dv.name} is selected")
        facility y_0_ is selected
        facility y_1_ is selected

        """  # noqa

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        fac = gdf_fac[facility_col]

        fac_type_geom = fac.geom_type.unique()

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(
                (
                    "Facility geodataframe contains mixed type geometries "
                    "or is not a point. Be sure deriving centroid from "
                    "geometries doesn't affect the results."
                ),
                UserWarning,
            )
            fac = fac.centroid

        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = cdist(fac_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances, p_facilities, predefined_facilities_arr, name
        )

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``PDispersion`` model.

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool (default True)
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.p_dispersion.PDispersion

        """
        self.problem.solve(solver)
        self.check_status()

        return self

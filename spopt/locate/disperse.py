import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import (
    BaseOutputMixin,
    CoveragePercentageMixin,
    LocateSolver,
    FacilityModelBuilder,
)
from scipy.spatial.distance import cdist

import warnings


class PDispersion(LocateSolver, BaseOutputMixin):
    """
    PDispersion class implements the p-dispersion optimization model and solves it.

    Parameters
    ----------
    name: str
        Problem name
    problem: pulp.LpProblem
        Pulp instance of optimization model that contains constraints, variables and objective function.

    Attributes
    ----------
    name: str
        Problem name
    problem: pulp.LpProblem
        Pulp instance of optimization model that contains constraints, variables and objective function.
    aij: np.array
        Cost matrix 2-d array containing the distances to each pair of facilities.
    """

    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add objective function to model:
        Maximize D

        Returns
        -------
        None
        """
        # Add Maximized Minimum Variable
        D = getattr(self, "D") #must add 'D' as an attribute to the class?
        self.problem += pulp.LpVariable('D', lowBound=0), "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        predefined_facilities_arr: np.array = None,
        name: str = "P-Dispersion",
    ):
        """
        Create a PDispersion object based on a cost matrix.

        Parameters
        ----------
        cost_matrix: np.array
            two-dimensional distance array between facility points.
        name: str, default="PDispersion"
            name of the problem

        Returns
        -------
        PDispersion object

        Examples
        --------
        >>> from spopt.locate.coverage import PDispersion
        >>> from spopt.locate.util import simulated_geo_points
        >>> import pulp
        >>> import spaghetti

        Create regular lattice

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> street = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> street_buffered = geopandas.GeoDataFrame(
        ...                            geopandas.GeoSeries(street["geometry"].buffer(0.2).unary_union),
        ...                            crs=street.crs,
        ...                            columns=["geometry"])

        Simulate points belong to lattice

        >>> facility_points = simulated_geo_points(street_buffered, needed=5, seed=6)

        Snap points to the network

        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(ntw, pp_name="facilities", snapped=True)

        Calculate the cost matrix
        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["facilities"],
        ...    destpattern=ntw.pointpatterns["facilities"])

        Create PDispersion instance from cost matrix

        >>> pDispersion_from_cost_matrix = PDispersion.from_cost_matrix(cost_matrix)
        >>> pDispersion_from_cost_matrix = PDispersion_from_cost_matrix.solve(pulp.PULP_CBC_CMD(msg=False))

        """

        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMaximize)
        pDispersion = PDispersion(name, model)

        FacilityModelBuilder.add_facility_integer_variable(pDispersion, r_fac, "y[{i}]")

        FacilityModelBuilder.add_maximized_min_variable(pDispersion,'D')

        pDispersion.aij = np.zeros(cost_matrix.shape)

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                pDispersion, pDispersion.problem, predefined_facilities_arr
            )

        pDispersion.__add_obj()
        FacilityModelBuilder.add_set_covering_constraint( #this will be some other covering constraint
            pDispersion, pDispersion.problem, pDispersion.aij, r_fac,
        )

        return pDispersion

    @classmethod
    def from_geodataframe(
        cls,
        gdf_fac: GeoDataFrame,
        facility_col: str,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "P-Dispersion",
    ):
        """
        Create a PDispersion object based on geodataframes. Calculate the cost matrix between facility and facility,
        and then use from_cost_matrix method.

        Parameters
        ----------
        gdf_fac: geopandas.GeoDataframe
            facility geodataframe with point geometry
        facility_col: str
            facility candidate sites geometry column name
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for the distance calculations
        name: str, default="P-Dispersion"
            name of the problem

        Returns
        -------
        PDispersion object

        Examples
        --------
        >>> from spopt.locate.coverage import PDispersion
        >>> from spopt.locate.util import simulated_geo_points
        >>> import pulp
        >>> import spaghetti

        Create regular lattice

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> street = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> street_buffered = geopandas.GeoDataFrame(
        ...                            geopandas.GeoSeries(street["geometry"].buffer(0.2).unary_union),
        ...                            crs=street.crs,
        ...                            columns=["geometry"])

        Simulate points belong to lattice

        >>> facility_points = simulated_geo_points(street_buffered, needed=5, seed=6)

        Snap points to the network

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(ntw, pp_name="facilities", snapped=True)

        Create PDispersion instance from cost matrix

        >>> pDispersion_from_geodataframe = PDispersion.from_geodataframe(facilities_snapped,
        ...                                                "geometry",
        ...                                                 distance_metric="euclidean")
        >>> pDispersion_from_geodataframe = pDispersion_from_geodataframe.solve(pulp.PULP_CBC_CMD(msg=False))

        """

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        fac = gdf_fac[facility_col]

        fac_type_geom = fac.geom_type.unique()

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(
                "Facility geodataframe contains mixed type geometries or is not a point. Be sure deriving centroid from geometries doesn't affect the results.",
                Warning,
            )
            fac = fac.centroid

        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = cdist(fac_data, fac_data, distance_metric) #altered, not sure if correct

        return cls.from_cost_matrix(
            distances, predefined_facilities_arr, name
        )

    def facility_client_array(self) -> None: # shouldn't need this
        """
        Create an array 2d MxN, where m is number of facilities and n is number of clients. Each row represent a facility and has an array containing clients index meaning that the facility-i cover the entire array.

        Returns
        -------
        None
        """

        fac_vars = getattr(self, "fac_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(self.aij.shape[0]):
                    if self.aij[i][j] > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the LSCP model

        Parameters
        ----------
        solver: pulp.LpSolver
            solver supported by pulp package

        results: bool
            if True it will create metainfo - which facilities cover which demand and vice-versa, and the uncovered demand - about the model results

        Returns
        -------
        LSCP object
        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()

        return self

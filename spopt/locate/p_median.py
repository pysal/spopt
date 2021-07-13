import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist


class PMedian(LocateSolver):
    """
    PMedian class implements P-Median optimization model and solve it.

    Parameters
    ----------
    name: str
        problem name
    problem: pulp.LpProblem
        pulp instance of optimization model that contains constraints, variables and objective function.
    sij: np.array
        two-dimensional array product of service load/population demand and distance matrix between facility and demand.

    """

    def __init__(self, name: str, problem: pulp.LpProblem, sij: np.array):
        self.sij = sij
        self.name = name
        self.problem = problem

    def __add_obj(self, range_clients: range, range_facility: range) -> None:
        """
        Add objective function to model
        Minimize s1_1 * z1_1 + s1_2 * z1_2 + ... + si_j * zi_j

        Parameters
        ----------
        range_clients: range
            range of demand points quantity
        range_facility: range
            range of demand facility quantity

        Returns
        -------
        None
        """
        cli_assgn_vars = getattr(self, "cli_assgn_vars")

        self.problem += (
            pulp.lpSum(
                [
                    self.sij[i][j] * cli_assgn_vars[i][j]
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
        ai: np.array,
        p_facilities: int,
        name: str = "p-median",
    ):
        """
        Create PMedian object based on cost matrix

        Parameters
        ----------
        cost_matrix: np.array
            two-dimensional distance array between facility points and demand point
        ai: np.array
            one-dimensional service load or population demand
        p_facilities: int
            number of facilities to be located
        name: str, default="p-median"
            name of the problem

        Returns
        -------
        PMedian object
        """
        r_cli = range(cost_matrix.shape[0])
        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMinimize)

        ai = np.reshape(ai, (cost_matrix.shape[0], 1))
        sij = ai * cost_matrix

        p_median = PMedian(name, model, sij)

        FacilityModelBuilder.add_facility_integer_variable(p_median, r_fac, "y[{i}]")
        FacilityModelBuilder.add_client_assign_integer_variable(
            p_median, r_cli, r_fac, "z[{i}_{j}]"
        )

        p_median.__add_obj(r_cli, r_fac)

        FacilityModelBuilder.add_facility_constraint(
            p_median, p_median.problem, p_facilities
        )
        FacilityModelBuilder.add_assignment_constraint(
            p_median, p_median.problem, r_fac, r_cli
        )
        FacilityModelBuilder.add_opening_constraint(
            p_median, p_median.problem, r_fac, r_cli
        )

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
        distance_metric: str = "euclidean",
        name: str = "p-median",
    ):
        """
        Create a PMedian object based on geodataframes. Calculate the cost matrix between demand and facility,
        and then use from_cost_matrix method.

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
        weights_cols: str
            weight column name representing service load or demand
        p_facilities: int
            number of facilities to be located
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for the distance calculations
        name: str, default="p-median"
            name of the problem

        Returns
        -------
        PMedian object
        """

        service_load = gdf_demand[weights_cols].to_numpy()
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = np.array([])

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(distances, service_load, p_facilities, name)

    def solve(self, solver: pulp.LpSolver):
        """
        Solve the PMedian model

        Parameters
        ----------
        solver: pulp.LpSolver
            solver supported by pulp package

        Returns
        -------
        PMedian object
        """
        self.problem.solve(solver)
        return self

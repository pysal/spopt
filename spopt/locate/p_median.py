import numpy as np

import pulp
from geopandas import GeoDataFrame

import spopt.locate
from spopt.locate.base import LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist


class PMedian(LocateSolver):
    def __init__(
        self, name: str, problem: pulp.LpProblem, zij: np.array, sij: np.array
    ):
        self.zij = zij
        self.sij = sij
        self.name = name
        self.problem = problem

    def __add_obj(self, range_clients: range, range_facility: range) -> None:
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
        cls, cost_matrix: np.array, ai: np.array, max_coverage: float, p_facilities: int
    ):
        r_cli = range(cost_matrix.shape[0])
        r_fac = range(cost_matrix.shape[1])

        name = "p-median"
        model = pulp.LpProblem(name, pulp.LpMinimize)

        aij = np.zeros(cost_matrix.shape)
        aij[cost_matrix <= max_coverage] = 1.0

        ai = np.reshape(ai, (cost_matrix.shape[0], 1))
        sij = ai * cost_matrix

        p_median = PMedian(name, model, aij, sij)

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
        max_coverage: float,
        p_facilities: int,
        distance_metric: str = "euclidean",
    ):

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

        return cls.from_cost_matrix(distances, service_load, max_coverage, p_facilities)

    def solve(self, solver: pulp.LpSolver):
        self.problem.solve(solver)

        if self.problem.status == pulp.constants.LpStatusUnbounded:
            raise Exception("unbounded solution")
        elif self.problem.status == pulp.constants.LpStatusNotSolved:
            raise Exception("not possible to solve")
        elif self.problem.status == pulp.constants.LpSolutionInfeasible:
            raise Exception("infeasible solution")
        elif self.problem.status == pulp.constants.LpSolutionOptimal:
            return self

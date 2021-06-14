import numpy as np

from spopt.locate.base import LocateSolver
import pulp
from geopandas import GeoDataFrame
from abc import ABC, abstractmethod
from spopt.locate.base import LocateSolver, FacilityModelBuilder


class LSCP(LocateSolver):
    def __init__(self, name: str, problem: pulp.LpProblem):
        self.name = name
        self.problem = problem

    def __add_obj(self):
        fac_vars = getattr(self, "fac_vars")
        self.problem += pulp.lpSum(fac_vars), "objective function"

    @classmethod
    def from_cost_matrix(cls, cost_matrix: np.array, max_coverage: float):
        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        name = "LSCP"
        model = pulp.LpProblem(name, pulp.LpMinimize)
        lscp = LSCP(name, model)

        FacilityModelBuilder.add_facility_integer_variable(lscp, r_fac)

        aij = np.zeros(cost_matrix.shape)
        aij[cost_matrix <= max_coverage] = 1

        FacilityModelBuilder.add_set_covering_constraint(
            lscp, lscp.problem, aij, r_fac, r_cli
        )
        lscp.__add_obj()

        return lscp

    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        max_coverage: float,
        distance_metric: str = "euclidean",
    ):
        distances = np.array()
        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        if distance_metric == "manhattan":
            # TODO
            pass
        elif distance_metric == "euclidean":
            distances = gdf_demand[demand_col].apply(
                lambda geometry: gdf_fac[facility_col].distance(geometry)
            )
            distances = distances.to_numpy()
        elif distance_metric == "haversine":
            pass
        else:
            raise ValueError("distance metric is not supported")

        return LSCP.from_cost_matrix(distances, max_coverage)

    def solve(self, solver: pulp.LpSolver):
        self.problem.solve(solver)

        if self.problem.status == pulp.constants.LpStatusUnbounded:
            raise Exception("unbounded solution")
        elif self.problem.status == pulp.constants.LpStatusNotSolved:
            raise Exception("not possible to solve")
        elif self.problem.status == pulp.constants.LpSolutionInfeasible:
            raise Exception("infeasible solution")


class MCLP(LocateSolver):
    def __init__(self, name: str, problem: pulp.LpProblem):
        self.name = name
        self.problem = problem

    def __add_obj(self, ai: np.array, range_clients: range):
        dem_vars = getattr(self, "cli_vars")
        self.problem += (
            pulp.lpSum([ai.flatten()[i] * dem_vars[i] for i in range_clients]),
            "objective function",
        )

    @classmethod
    def from_cost_matrix(
        cls, cost_matrix: np.array, ai: np.array, max_coverage: float, p_facilities: int
    ):
        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        name = "MCLP"
        model = pulp.LpProblem(name, pulp.LpMaximize)
        mclp = MCLP(name, model)

        FacilityModelBuilder.add_facility_integer_variable(mclp, r_fac)

        aij = np.zeros(cost_matrix.shape)
        aij[cost_matrix <= max_coverage] = 1

        FacilityModelBuilder.add_set_covering_constraint(
            mclp, mclp.problem, aij, r_fac, r_cli
        )
        FacilityModelBuilder.add_facility_constraint(mclp, mclp.problem, p_facilities)
        mclp.__add_obj(ai, r_cli)

        return mclp

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
        service_load = gdf_fac[weights_cols].to_numpy()
        distances = np.array()

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        if distance_metric == "manhattan":
            # TODO
            pass
        elif distance_metric == "euclidean":
            distances = gdf_demand[demand_col].apply(
                lambda geometry: gdf_fac[facility_col].distance(geometry)
            )
            distances = distances.to_numpy()
        elif distance_metric == "haversine":
            pass
        else:
            raise ValueError("distance metric is not supported")

        return MCLP.from_cost_matrix(
            distances, service_load, max_coverage, p_facilities
        )

    def solve(self, solver: pulp.LpSolver):
        self.problem.solve(solver)

        if self.problem.status == pulp.constants.LpStatusUnbounded:
            raise Exception("unbounded solution")
        elif self.problem.status == pulp.constants.LpStatusNotSolved:
            raise Exception("not possible to solve")
        elif self.problem.status == pulp.constants.LpSolutionInfeasible:
            raise Exception("infeasible solution")

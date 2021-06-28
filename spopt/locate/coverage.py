import numpy as np

from spopt.locate.base import LocateSolver
import pulp
from geopandas import GeoDataFrame
from abc import ABC, abstractmethod
from spopt.locate.base import LocateSolver, FacilityModelBuilder
from scipy.spatial import distance_matrix


class Coverage:
    def __init__(self, name: str, problem: pulp.LpProblem):
        self.name = name
        self.problem = problem
        self.aij = np.array([[]])

    def uncovered_clients_dict(self):
        self.n_cli_uncov = self.aij.shape[0] - len(self.cli2iloc.keys())

    def client_facility_dict(self):
        self.cli2fac = {}
        for cv in list(self.cli2iloc.keys()):
            self.cli2fac[cv] = []
            for k, v in self.fac2cli.items():
                if cv in v:
                    self.cli2fac[cv].append(k)

    def cov_dict(self):
        self.cli2ncov = {}
        for c, fs in self.cli2fac.items():
            self.cli2ncov[c] = len(fs)
            most_coverage = max(self.cli2ncov.values())
            self.ncov2ncli = {}
            for cov_count in range(most_coverage + 1):
                if cov_count == 0:
                    self.ncov2ncli[cov_count] = self.n_cli_uncov
                    continue
                if not cov_count in list(self.cli2ncov.keys()):
                    self.ncov2ncli[cov_count] = 0
                for c, ncov in self.cli2ncov.items():
                    if ncov >= cov_count:
                        self.ncov2ncli[cov_count] += 1


class LSCP(LocateSolver, Coverage):
    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

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

        FacilityModelBuilder.add_facility_integer_variable(lscp, r_fac, "x[{i}]")

        lscp.aij = np.zeros(cost_matrix.shape)
        lscp.aij[cost_matrix <= max_coverage] = 1

        lscp.__add_obj()
        FacilityModelBuilder.add_set_covering_constraint(
            lscp, lscp.problem, lscp.aij, r_fac, r_cli
        )

        return lscp

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        max_coverage: float,
        distance_metric: str = "euclidean",
    ):
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = np.array([])

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        if distance_metric == "manhattan":
            distances = distance_matrix(dem_data, fac_data, p=1)
        elif distance_metric == "euclidean":
            distances = distance_matrix(dem_data, fac_data, p=2)
        else:
            raise ValueError("distance metric is not supported")

        return cls.from_cost_matrix(distances, max_coverage)

    def record_decisions(self):
        fac_vars = getattr(self, "fac_vars")
        self.cli2iloc = {}
        self.fac2cli = {}

        for j in range(len(fac_vars)):
            if fac_vars[j].value() > 0:
                fac_var_name = fac_vars[j].name
                self.fac2cli[fac_var_name] = []
                for i in range(self.aij.shape[0]):
                    if self.aij[i][j] > 0:
                        cli_var_name = f"N[{i}]"
                        self.fac2cli[fac_var_name].append(cli_var_name)
                        self.cli2iloc[cli_var_name] = i

        self.client_facility_dict()
        self.uncovered_clients_dict()
        self.cov_dict()

    def solve(self, solver: pulp.LpSolver):
        self.problem.solve(solver)

        if self.problem.status == pulp.constants.LpStatusUnbounded:
            raise Exception("unbounded solution")
        elif self.problem.status == pulp.constants.LpStatusNotSolved:
            raise Exception("not possible to solve")
        elif self.problem.status == pulp.constants.LpSolutionInfeasible:
            raise Exception("infeasible solution")
        elif self.problem.status == pulp.constants.LpSolutionOptimal:
            return 1


class MCLP(LocateSolver, Coverage):
    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

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

        FacilityModelBuilder.add_facility_integer_variable(mclp, r_fac, "x[{i}]")
        FacilityModelBuilder.add_client_integer_variable(mclp, r_cli, "y[{i}]")

        mclp.aij = np.zeros(cost_matrix.shape)
        mclp.aij[cost_matrix <= max_coverage] = 1

        mclp.__add_obj(ai, r_cli)
        FacilityModelBuilder.add_maximal_coverage_constraint(
            mclp, mclp.problem, mclp.aij, r_fac, r_cli
        )
        FacilityModelBuilder.add_facility_constraint(mclp, mclp.problem, p_facilities)

        return mclp

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

        if distance_metric == "manhattan":
            distances = distance_matrix(dem_data, fac_data, p=1)
        elif distance_metric == "euclidean":
            distances = distance_matrix(dem_data, fac_data, p=2)
        else:
            raise ValueError("distance metric is not supported")

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
            return 1

    def record_decisions(self):
        fac_vars = getattr(self, "fac_vars")
        cli_vars = getattr(self, "cli_vars")
        self.cli2iloc = {}
        self.fac2cli = {}

        for j in range(len(fac_vars)):
            if fac_vars[j].value() > 0:
                fac_var_name = fac_vars[j].name
                self.fac2cli[fac_var_name] = []
                for i in range(self.aij.shape[0]):
                    if cli_vars[i].value() > 0:
                        if self.aij[i][j] > 0:
                            cli_var_name = cli_vars[i].name
                            self.fac2cli[fac_var_name].append(cli_var_name)
                            self.cli2iloc[cli_var_name] = i

        self.client_facility_dict()
        self.uncovered_clients_dict()
        self.cov_dict()

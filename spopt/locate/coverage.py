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


class LSCP(LocateSolver, BaseOutputMixin):
    """
    LSCP class implements Location Set Covering optimization model and solve it.

    Parameters
    ----------
    name: str
        Problem name
    problem: pulp.LpProblem
        Pulp instance of optimization model that contains constraints, variables and objective function.

    """

    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add objective function to model:
        Minimize x1 + x2 + x3 + x4 + x5 + ... + xj

        Returns
        -------
        None
        """
        fac_vars = getattr(self, "fac_vars")
        self.problem += pulp.lpSum(fac_vars), "objective function"

    @classmethod
    def from_cost_matrix(
        cls, cost_matrix: np.array, max_coverage: float, name: str = "LSCP"
    ):
        """
        Create a LSCP object based on cost matrix.

        Parameters
        ----------
        cost_matrix: np.array
            two-dimensional distance array between facility points and demand point
        max_coverage: float
            maximum acceptable service distance by problem
        name: str, default="LSCP"
            name of the problem

        Returns
        -------
        LSCP object
        """

        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

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
        name: str = "LSCP",
    ):
        """
        Create a LSCP object based on geodataframes. Calculate the cost matrix between demand and facility,
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
        max_coverage: float
            maximum acceptable service distance by problem
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for the distance calculations
        name: str, default="LSCP"
            name of the problem

        Returns
        -------
        LSCP object
        """

        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(
                "Demand geodataframe contains mixed type geometries or is not a point. Be sure deriving centroid from geometries doesn't affect the results.",
                Warning,
            )
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(
                "Facility geodataframe contains mixed type geometries or is not a point. Be sure deriving centroid from geometries doesn't affect the results.",
                Warning,
            )
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = np.array([])

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(distances, max_coverage, name)

    def facility_client_array(self) -> None:
        """
        Create an array 2d $m$ x $n$, where m is number of facilities and n is number of clients. Each row represent a facility and has an array containing clients index meaning that the $facility_0$ cover the entire array.

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

    def solve(self, solver: pulp.LpSolver):
        """
        Solve the LSCP model

        Parameters
        ----------
        solver: pulp.LpSolver
            solver supported by pulp package

        Returns
        -------
        LSCP object
        """
        self.problem.solve(solver)
        return self


class MCLP(LocateSolver, BaseOutputMixin, CoveragePercentageMixin):
    """
    MCLP class implements Maximal Coverage Location optimization model and solve it.

    Parameters
    ----------
    name: str
        Problem name
    problem: pulp.LpProblem
        Pulp instance of optimization model that contains constraints, variables and objective function.

    """

    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

    def __add_obj(self, weights: np.array, range_clients: range) -> None:
        """
        Add objective function to model:
        Maximize w1 * y1 + w2 * y2 +  ... + wi * yi

        Returns
        -------
        None
        """
        dem_vars = getattr(self, "cli_vars")

        self.problem += (
            pulp.lpSum([weights.flatten()[i] * dem_vars[i] for i in range_clients]),
            "objective function",
        )

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        weights: np.array,
        max_coverage: float,
        p_facilities: int,
        name: str = "MCLP",
    ):
        """
        Create a MCLP object based on cost matrix.

        Parameters
        ----------
        cost_matrix: np.array
            two-dimensional distance array between facility points and demand point
        weights: np.array
            one-dimensional service load or population demand
        max_coverage: float
            maximum acceptable service distance by problem
        p_facilities: int
            number of facilities to be located
        name: str, default="MCLP"
            name of the problem

        Returns
        -------
        MCLP object
        """
        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        model = pulp.LpProblem(name, pulp.LpMaximize)
        mclp = MCLP(name, model)

        FacilityModelBuilder.add_facility_integer_variable(mclp, r_fac, "x[{i}]")
        FacilityModelBuilder.add_client_integer_variable(mclp, r_cli, "y[{i}]")

        mclp.aij = np.zeros(cost_matrix.shape)
        mclp.aij[cost_matrix <= max_coverage] = 1
        weights = np.reshape(weights, (cost_matrix.shape[0], 1))

        mclp.__add_obj(weights, r_cli)
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
        name: str = "MCLP",
    ):
        """
        Create a MCLP object based on geodataframes. Calculate the cost matrix between demand and facility,
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
        max_coverage: float
            maximum acceptable service distance by problem
        p_facilities: int
            number of facilities to be located
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for the distance calculations
        name: str, default="MCLP"
            name of the problem

        Returns
        -------
        MCLP object
        """
        service_load = gdf_demand[weights_cols].to_numpy()
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(
                "Demand geodataframe contains mixed type geometries or is not a point. Be sure deriving centroid from geometries doesn't affect the results.",
                Warning,
            )
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(
                "Facility geodataframe contains mixed type geometries or is not a point. Be sure deriving centroid from geometries doesn't affect the results.",
                Warning,
            )
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = np.array([])

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                f"geodataframes crs are different: gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances, service_load, max_coverage, p_facilities, name
        )

    def facility_client_array(self) -> None:
        """
        Create an array 2d $m$ x $n$, where m is number of facilities and n is number of clients. Each row represent a facility and has an array containing clients index meaning that the $facility_0$ cover the entire array.

        Returns
        -------
        None
        """

        fac_vars = getattr(self, "fac_vars")
        cli_vars = getattr(self, "cli_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(self.aij.shape[0]):
                    if cli_vars[i].value() > 0:
                        if self.aij[i][j] > 0:
                            array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver):
        """
        Solve the MCLP model

        Parameters
        ----------
        solver: pulp.LpSolver
            solver supported by pulp package

        Returns
        -------
        MCLP object
        """
        self.problem.solve(solver)
        return self

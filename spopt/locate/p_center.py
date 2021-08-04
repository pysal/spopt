import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import BaseOutputMixin, LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist

import warnings


class PCenter(LocateSolver, BaseOutputMixin):
    """
    PCenter class implements P-Center optimization model and solve it.

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
        self.problem = problem
        self.name = name
        self.sij = sij

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
        weights: np.array,
        p_facilities: int,
        name: str = "p-center",
    ):
        """
        Create PCenter object based on cost matrix

        Parameters
        ----------
        cost_matrix: np.array
            two-dimensional distance array between facility points and demand point
        weights: np.array
            one-dimensional service load or population demand
        p_facilities: int
            number of facilities to be located
        name: str, default="p-center"
            name of the problem

        Returns
        -------
        PCenter object
        """
        r_cli = range(cost_matrix.shape[0])
        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMinimize)

        weights = np.reshape(weights, (cost_matrix.shape[0], 1))
        sij = weights * cost_matrix

        p_center = PCenter(name, model, sij)

        FacilityModelBuilder.add_facility_integer_variable(p_center, r_fac, "y[{i}]")
        FacilityModelBuilder.add_client_assign_integer_variable(
            p_center, r_cli, r_fac, "z[{i}_{j}]"
        )
        FacilityModelBuilder.add_weight_continuous_variable(p_center)

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
        weights_cols: str,
        p_facilities: int,
        distance_metric: str = "euclidean",
        name: str = "p-center",
    ):
        """
        Create a PCenter object based on geodataframes. Calculate the cost matrix between demand and facility,
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
        PCenter object
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

        return cls.from_cost_matrix(distances, service_load, p_facilities, name)

    def facility_client_array(self) -> None:
        """
        Create an array 2d $m$ x $n$, where m is number of facilities and n is number of clients. Each row represent a facility and has an array containing clients index meaning that the $facility_0$ cover the entire array.

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

    def solve(self, solver: pulp.LpSolver):
        """
        Solve the PCenter model

        Parameters
        ----------
        solver: pulp.LpSolver
            solver supported by pulp package

        Returns
        -------
        PCenter object
        """
        self.problem.solve(solver)
        return self

from abc import ABC, abstractmethod, abstractproperty
import pulp
from geopandas import GeoDataFrame
import numpy as np
from typing import Type, TypeVar


class LocateSolver(ABC):
    @abstractmethod
    def solve(self, solver: pulp.LpSolver):
        pass


T_FacModel = TypeVar("T_FacModel", bound=LocateSolver)


class FacilityModelBuilder(object):
    """
    Set Constraints and Variables
    """

    @staticmethod
    def add_facility_integer_variable(obj: T_FacModel, range_facility):
        fac_vars = [
            pulp.LpVariable(f"x[{i}]", lowBound=0, upBound=1, cat=pulp.LpInteger)
            for i in range_facility
        ]

        setattr(obj, "fac_vars", fac_vars)

    @staticmethod
    def add_client_integer_variable(obj: T_FacModel, range_client):
        cli_vars = [
            pulp.LpVariable(f"y[{i}]", lowBound=0, upBound=1, cat="Integer")
            for i in range_client
        ]

        setattr(obj, "cli_vars", cli_vars)

    @staticmethod
    def add_set_covering_constraint(
        obj: T_FacModel, model, ni, range_facility, range_client
    ):
        try:
            fac_vars = getattr(obj, "fac_vars")
            for i in range_facility:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_client]) >= 1,
                    "set covering constraint",
                )
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

    @staticmethod
    def add_facility_constraint(obj: T_FacModel, model, p_facilities):
        try:
            fac_vars = getattr(obj, "fac_vars")
            model += pulp.lpSum(fac_vars) == p_facilities
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

    @staticmethod
    def add_maximal_coverage_constraint(
        obj: T_FacModel, model, ni, range_facility, range_client
    ):
        try:
            fac_vars = getattr(obj, "fac_vars")
            dem_vars = getattr(obj, "cli_vars")
            for i in range_client:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_facility])
                    >= dem_vars[i]
                )
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

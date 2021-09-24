from abc import abstractmethod

from spopt.BaseClass import BaseSpOptSolver
from typing import TypeVar

import numpy as np
import pulp


class LocateSolver(BaseSpOptSolver):
    """
    Base Class for locate package
    """

    def __init__(self, name: str, problem: pulp.LpProblem):
        self.name = name
        self.problem = problem
        self.aij = np.array([[]])

    @abstractmethod
    def solve(self, solver: pulp.LpSolver):
        """
        Solve the optimization model

        Parameters
        ----------
        solver: pulp.LpSolver
                solver supported by pulp package

        Returns
        -------
        None
        """
        pass


class BaseOutputMixin:
    """
    Base Mixin used by all models
    """

    def client_facility_array(self) -> None:
        """
        Create an array 2d MxN, where m is number of clients and n is number of facilities.
        """
        if hasattr(self, "fac2cli"):
            self.cli2fac = [[] for i in range(self.aij.shape[0])]

            for i in range(len(self.fac2cli)):
                for fac_site in self.fac2cli[i]:
                    self.cli2fac[fac_site].append(i)

    def uncovered_clients(self) -> None:
        """
        Calculate how many clients points are not covered
        """
        set_cov = set()
        for i in range(len(self.fac2cli)):
            set_cov |= set(self.fac2cli[i])

        self.n_cli_uncov = self.aij.shape[0] - len(set_cov)


class CoveragePercentageMixin:
    """
    Mixin to calculate the percentage of area covered
    """

    def get_percentage(self):
        """
        Calculate the percentage
        """
        self.percentage = 1 - (self.n_cli_uncov / self.aij.shape[0])


class MeanDistanceMixin:
    """
    Mixin to calculate the mean distance between demand and facility sites chosen
    """

    def get_mean_distance(self, weight: np.array):
        """
        Calculate the mean distance
        Parameters
        ----------
        weight: np.array
                weight of all demand points

        Returns
        -------
        None
        """
        self.mean_dist = self.problem.objective.value() / weight.sum()


T_FacModel = TypeVar("T_FacModel", bound=LocateSolver)


class FacilityModelBuilder:
    """
    Set facility locations' variables and constraints
    """

    @staticmethod
    def add_facility_integer_variable(
        obj: T_FacModel, range_facility, var_name
    ) -> None:
        """

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        range_facility: range
            range of facility points quantity
        var_name: str
            formatted string
            facility variable name

        Returns
        -------
        None
        """
        fac_vars = [
            pulp.LpVariable(
                var_name.format(i=i), lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            for i in range_facility
        ]

        setattr(obj, "fac_vars", fac_vars)

    @staticmethod
    def add_client_integer_variable(obj: T_FacModel, range_client, var_name) -> None:
        """

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        range_client: range
            range of demand points quantity
        var_name: str
            formatted string
            demand variable name

        Returns
        -------
        None
        """
        cli_vars = [
            pulp.LpVariable(
                var_name.format(i=i), lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            for i in range_client
        ]

        setattr(obj, "cli_vars", cli_vars)

    @staticmethod
    def add_client_assign_integer_variable(
        obj: T_FacModel, range_client, range_facility, var_name
    ) -> None:
        """

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        range_client: range
            range of demand points quantity
        range_facility: range
            range of facility points quantity
        var_name: str
            formatted string
            client assigning variable name

        Returns
        -------
        None
        """
        cli_assgn_vars = [
            [
                pulp.LpVariable(
                    var_name.format(i=i, j=j), lowBound=0, upBound=1, cat=pulp.LpInteger
                )
                for j in range_facility
            ]
            for i in range_client
        ]

        setattr(obj, "cli_assgn_vars", cli_assgn_vars)

    @staticmethod
    def add_weight_continuous_variable(obj: T_FacModel) -> None:
        """

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class

        Returns
        -------
        None
        """
        weight_var = pulp.LpVariable("W", lowBound=0, cat=pulp.LpContinuous)

        setattr(obj, "weight_var", weight_var)

    @staticmethod
    def add_set_covering_constraint(
        obj: T_FacModel,
        model: pulp.LpProblem,
        ni: np.array,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        set covering constraint:
        n1_1 * fac_var1 + n1_2 * fac_var1 + ... + nij * fac_varj >= 1

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        ni: np.array
            two-dimensional array that defines candidate sites between facility points within a distance to supply {i}
            demand point
        range_facility: range
            range of facility points quantity
        range_client: range
            range of demand points quantity
        Returns
        -------
        None

        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            for i in range_client:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_facility]) >= 1
                )
        else:
            raise AttributeError(
                "before setting constraints must set facility variable"
            )

    @staticmethod
    def add_facility_constraint(
        obj: T_FacModel, model: pulp.LpProblem, p_facilities: int
    ) -> None:
        """
        set facility constraint:
        fac_var1 + fac_var2 + fac_var3 + ... + fac_varj == p

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        p_facilities: int
            maximum number of facilities can be used

        Returns
        -------
        None
        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            model += pulp.lpSum(fac_vars) == p_facilities
        else:
            raise AttributeError(
                "before setting constraints must set facility variable"
            )

    @staticmethod
    def add_maximal_coverage_constraint(
        obj: T_FacModel, model, ni, range_facility, range_client
    ) -> None:
        """
        set maximal constraint:
        n1_1 * fac_var1 + n1_2 * fac_var1 + ... + nij * fac_varj >= dem_var[i]

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        ni: np.array
            two-dimensional array that defines candidate sites between facility points within a distance to supply {i} demand point
        range_facility: range
            range of facility points quantity
        range_client: range
            range of demand points quantity

        Returns
        -------
        None
        """
        if hasattr(obj, "fac_vars") and hasattr(obj, "cli_vars"):
            fac_vars = getattr(obj, "fac_vars")
            dem_vars = getattr(obj, "cli_vars")
            for i in range_client:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_facility])
                    >= dem_vars[i]
                )
        else:
            raise AttributeError(
                "before setting constraints must set facility and demand variable"
            )

    @staticmethod
    def add_assignment_constraint(
        obj: T_FacModel, model, range_facility, range_client
    ) -> None:
        """
        set assignment constraint:
        x1_1 + x_1_2 + x1_3 + x1_j == 1

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        range_facility: range
            range of facility points quantity
        range_client: range
            range of demand points quantity

        Returns
        -------
        None
        """
        if hasattr(obj, "cli_assgn_vars"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")

            for i in range_client:
                model += pulp.lpSum([cli_assgn_vars[i][j] for j in range_facility]) == 1
        else:
            raise AttributeError(
                "before setting constraints must set client assignment variable"
            )

    @staticmethod
    def add_opening_constraint(
        obj: T_FacModel, model, range_facility, range_client
    ) -> None:
        """
        set opening constraint to model:
        fac_var_j >= xi_j

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        range_facility: range
            range of facility points quantity
        range_client: range
            range of demand points quantity

        Returns
        -------
        None
        """
        if hasattr(obj, "cli_assgn_vars"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            fac_vars = getattr(obj, "fac_vars")

            for i in range_client:
                for j in range_facility:
                    model += fac_vars[j] - cli_assgn_vars[i][j] >= 0
        else:
            raise AttributeError(
                "before setting constraints must set client assignment variable"
            )

    @staticmethod
    def add_minimized_maximum_constraint(
        obj: T_FacModel, model, cost_matrix, range_facility, range_client
    ) -> None:
        """
        set minimized maximum constraint:
        x1_1 * d1_1 + x1_2 * d1_2 + ... + xi_j * di_j <= W

        Parameters
        ----------
        obj: T_FacModel
            bounded type of LocateSolver class
        model: pulp.LpProblem
            optimization model problem
        cost_matrix: np.array
            two-dimensional distance array between facility points and demand point
        range_facility: range
            range of facility points quantity
        range_client: range
            range of demand points quantity

        Returns
        -------
        None

        Notes
        -----
        See explanation W variable in ``spopt.locate.base.add_weight_continuous_variable``
        """
        if hasattr(obj, "cli_assgn_vars") and hasattr(obj, "weight_var"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            weight_var = getattr(obj, "weight_var")

            for i in range_client:
                model += (
                    pulp.lpSum(
                        [
                            cli_assgn_vars[i][j] * cost_matrix[i][j]
                            for j in range_facility
                        ]
                    )
                    <= weight_var
                )
        else:
            raise AttributeError(
                "before setting constraints must set weight and client assignment variables"
            )

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import pulp


class LocateSolver(ABC):
    """
    Base Class for locate package
    """

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


T_FacModel = TypeVar("T_FacModel", bound=LocateSolver)


class FacilityModelBuilder(object):
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
        try:
            fac_vars = getattr(obj, "fac_vars")
            for i in range_client:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_facility]) >= 1
                )
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

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
        try:
            fac_vars = getattr(obj, "fac_vars")
            model += pulp.lpSum(fac_vars) == p_facilities
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

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
        try:
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")

            for i in range_client:
                model += pulp.lpSum([cli_assgn_vars[i][j] for j in range_facility]) == 1
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

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
        try:
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            fac_vars = getattr(obj, "fac_vars")

            for i in range_client:
                for j in range_facility:
                    model += fac_vars[j] - cli_assgn_vars[i][j] >= 0
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

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
        try:
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
        except AttributeError:
            raise Exception("before setting constraints must set facility variable")

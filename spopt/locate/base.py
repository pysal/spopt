from abc import abstractmethod

from ..BaseClass import BaseSpOptExactSolver
from typing import TypeVar

import numpy as np
import pulp


# https://coin-or.github.io/pulp/technical/constants.html#pulp.constants.LpStatus
STATUS_CODES = {
    1: "Optimal",
    0: "Not Solved",
    -1: "Infeasible",
    -2: "Unbounded",
    -3: "Undefined",
}


class LocateSolver(BaseSpOptExactSolver):
    """Base class for the ``locate`` package."""

    def __init__(self, name: str, problem: pulp.LpProblem):
        self.name = name
        self.problem = problem
        self.aij = np.array([[]])

    @abstractmethod
    def solve(self, solver: pulp.LpSolver):
        """
        Solve the optimization model.

        Parameters
        ----------

        solver : pulp.apis.LpSolver
            A solver supported by ``pulp``.

        Returns
        -------

        None

        """
        pass

    def check_status(self):
        """Ensure a model is solved."""

        if self.problem.status != 1:
            status = STATUS_CODES[self.problem.status]
            msg = (
                f"Model is not solved: {status}. "
                "See ``pulp.constants.LpStatus`` for more information."
            )
            raise RuntimeError(msg)


class BaseOutputMixin:
    """Base Mixin used by all models with clients."""

    def client_facility_array(self) -> None:
        """

        Create a 2D array storing **client to facility relationships** where each
        row represents a client and contains an array of facility indices
        with which it is associated. An empty facility array indicates
        the client is associated with no facility.

        Notes
        -----

        This function requires ``fac2cli`` attribute to work properly.
        This attribute is set using ``facility_client_array`` method
        which is located inside the model classes. When the ``solve``
        method is called with ``results=True`` it will be called automatically,
        if not, you have to call the method manually post-solve.

        """
        if hasattr(self, "fac2cli"):
            self.cli2fac = [[] for i in range(self.aij.shape[0])]

            for i in range(len(self.fac2cli)):
                for fac_site in self.fac2cli[i]:
                    self.cli2fac[fac_site].append(i)
        else:
            raise AttributeError(
                "The attribute `fac2cli` is not set. "
                "See `facility_client_array` method to set the attribute"
            )


class CoveragePercentageMixin:
    """
    Mixin to calculate the percentage of area covered.

    Notes
    -----

    This Mixin requires the ``n_cli_uncov`` attribute.
    This attribute is set using the ``uncovered_clients`` method which is located
    inside the model classes. When the ``solve`` method is called with ``results=True``
    it will already set automatically, if not, you have to call the method.

    """

    def uncovered_clients(self) -> None:
        """
        Calculate how many clients points are not covered.

        Notes
        -----

        This method requires ``fac2cli`` attribute to work properly.
        This attribute is set using ``facility_client_array`` method
        which is located inside the model classes. When the ``solve``
        method is called with ``results=True`` it will be called automatically,
        if not, you have to call the method manually post-solve.

        """

        if hasattr(self, "fac2cli"):
            set_cov = set()
            for i in range(len(self.fac2cli)):
                set_cov |= set(self.fac2cli[i])

            self.n_cli_uncov = self.aij.shape[0] - len(set_cov)
        else:
            raise AttributeError(
                "The attribute `fac2cli` is not set. See `facility_client_array` "
                "method to set the attribute."
            )

    def get_percentage(self):
        """Calculate the percentage of covered clients."""
        if hasattr(self, "n_cli_uncov"):
            self.perc_cov = (1 - (self.n_cli_uncov / self.aij.shape[0])) * 100.0
        else:
            raise AttributeError(
                "The attribute `n_cli_uncov` is not set. See `uncovered_clients` "
                "method to set the attribute."
            )


class MeanDistanceMixin:
    """
    Mixin to calculate the mean distance between demand and selected facility sites.
    """

    def get_mean_distance(self):
        """Calculate the mean distance."""
        self.mean_dist = self.problem.objective.value() / self.ai_sum


class BackupPercentageMixinMixin:
    """
    Mixin to calculate the percentage of clients being covered by
    more the one facility (*LSCP-B*).
    """

    def get_percentage(self):
        """Calculate the percentage of clients with backup."""
        self.backup_perc = (self.problem.objective.value() / len(self.cli_vars)) * 100.0


T_FacModel = TypeVar("T_FacModel", bound=LocateSolver)


class FacilityModelBuilder:
    """Set facility location modeling variables and constraints."""

    @staticmethod
    def add_facility_integer_variable(
        obj: T_FacModel, range_facility: range, var_name: str
    ) -> None:
        """Facility integer decision variables.

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_facility : range
            The range of facility points.
        var_name : str
            A formatted string for the facility variable name.

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
    def add_client_integer_variable(
        obj: T_FacModel, range_client: range, var_name: str
    ) -> None:
        """Client integer decision variables.

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_client : range
            The range of demand points.
        var_name : str
            A formatted string for the demand variable name.

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
    def add_client_assign_variable(
        obj: T_FacModel,
        range_client: range,
        range_facility: range,
        var_name: str,
        low_bound=0,
        up_bound=1,
        lp_category=pulp.LpInteger,
    ) -> None:
        """Client assignment integer decision variables (used for allocation).

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_client : range
            The range of demand points.
        range_facility : range
            The range of facility points.
        var_name : str
            A formatted string for the  client assignment variable name.
        low_bound : int (default 0)
            The lower bound for variable values. Set to ``None`` for no lower bound.
        up_bound : int (default 1)
            The upper bound for variable values. Set to ``None`` for no upper bound.
        lp_category : pulp.LpVariable parameter
            The category this variable is in,
            ``pulp.LpInteger`` or ``pulp.LpContinuous``.

        Returns
        -------

        None

        """

        cli_assgn_vars = np.array(
            [
                [
                    pulp.LpVariable(
                        var_name.format(i=i, j=j),
                        lowBound=low_bound,
                        upBound=up_bound,
                        cat=lp_category,
                    )
                    for j in range_facility
                ]
                for i in range_client
            ]
        )

        setattr(obj, "cli_assgn_vars", cli_assgn_vars)

    @staticmethod
    def add_weight_continuous_variable(obj: T_FacModel) -> None:
        """Maximized minimum variable (p-center).

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.

        Returns
        -------

        None

        """
        weight_var = pulp.LpVariable("W", lowBound=0, cat=pulp.LpContinuous)

        setattr(obj, "weight_var", weight_var)

    @staticmethod
    def add_maximized_min_variable(obj: T_FacModel) -> None:
        """Maximized minimum variable (p-dispersion).

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.

        Returns
        -------

        None

        """
        D = pulp.LpVariable("D", lowBound=0, cat=pulp.LpContinuous)
        setattr(obj, "disperse_var", D)

    @staticmethod
    def add_set_covering_constraint(
        obj: T_FacModel,
        range_client: range,
        range_facility: range,
    ) -> None:
        """
        Create set covering constraints.

        ni0 * y0 + ni1 * y1 + ... + nij * yj >= 1

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_client : range
            The range of demand points.
        range_facility : range
            The range of facility points.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            model = getattr(obj, "problem")
            ni = getattr(obj, "aij")
            for i in range_client:
                model += (
                    pulp.lpSum([ni[i, j] * fac_vars[j] for j in range_facility]) >= 1
                )
        else:
            raise AttributeError(
                "Before setting coverage constraints facility variables must be set."
            )

    @staticmethod
    def add_backup_covering_constraint(
        obj: T_FacModel,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        Create backup covering constraints.

        - u_i + ni0 * y_0 + ni1 * y_1 + ... + nij * y_j >= 1

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_facility : range
            The range of facility points.
        range_client : range
            The range of demand points.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            cli_vars = getattr(obj, "cli_vars")
            model = getattr(obj, "problem")
            ni = getattr(obj, "aij")
            for i in range_client:
                if sum(ni[i]) >= 2:
                    model += (
                        pulp.lpSum(
                            [int(ni[i, j]) * fac_vars[j] for j in range_facility]
                        )
                        >= 1 + 1 * cli_vars[i]
                    )
                else:
                    model += (
                        pulp.lpSum(
                            [int(ni[i, j]) * fac_vars[j] for j in range_facility]
                        )
                        >= 1 + 0 * cli_vars[i]
                    )
        else:
            raise AttributeError(
                "Before setting backup coverage constraints "
                "facility variables must be set."
            )

    @staticmethod
    def add_facility_constraint(obj: T_FacModel, p_facilities: int) -> None:
        """
        Create the facility constraint.

        y0 + y1 + ... + yj == p

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        p_facilities : int
            The number of facilities to be sited.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            model = getattr(obj, "problem")
            model += pulp.lpSum(fac_vars) == p_facilities
        else:
            raise AttributeError(
                "Before setting facility constraint facility variables must be set."
            )

    @staticmethod
    def add_predefined_facility_constraint(
        obj: T_FacModel, predefined_fac: np.array
    ) -> None:
        """
        Create predefined demand constraints.

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        facility_indexes : numpy.array
            Indexes of facilities that are already located (zero-indexed).

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars"):
            fac_vars = getattr(obj, "fac_vars")
            model = getattr(obj, "problem")
            for ind in range(len(predefined_fac)):
                if predefined_fac[ind]:
                    fac_vars[ind].setInitialValue(1)
                    fac_vars[ind].fixValue()
        else:
            raise AttributeError(
                "Before setting predefined facility constraints "
                "facility variables must be set."
            )

    @staticmethod
    def add_facility_capacity_constraint(
        obj: T_FacModel,
        dq_ni: np.array,
        cl_ni: np.array,
        range_client: range,
        range_facility: range,
    ) -> None:
        """
        Create the facility capacity constraints:

        Demand at :math:`i` multiplied by the fraction of demand :math:`i`
        assigned to facility :math:`j` must be less than or equal to the
        capacity at facility :math:`j`.

        :math:`a_i Z_{ij} \leq C_j X_j`

        n1_1 * fac_var1 + n1_2 * fac_var1 + ... + nij * fac_varj >= dem_var[i]

        Parameters
        ----------

        obj: T_FacModel
            A bounded type of the ``LocateSolver`` class.
        dq_ni : numpy.array
            A 1D array that defines demand quantities for demand points.
        cl_ni : numpy.array
            A 1D array that defines capacity limits of facility points.
        range_client : range
            The range of demand points.
        range_facility : range
            The range of facility points.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars") and hasattr(obj, "cli_assgn_vars"):
            fac_vars = getattr(obj, "fac_vars")
            cli_assn_vars = getattr(obj, "cli_assgn_vars")
            model = getattr(obj, "problem")

            for j in range_facility:
                model += (
                    pulp.lpSum([dq_ni[i] * cli_assn_vars[i, j] for i in range_client])
                    <= cl_ni[j] * fac_vars[j]
                )
        else:
            raise AttributeError(
                "Facility and client assignment variables must "
                "be set prior to adding facility capacity constraints."
            )

    @staticmethod
    def add_client_demand_satisfaction_constraint(
        obj: T_FacModel,
        range_client: range,
        range_facility: range,
    ) -> None:
        """
        Create the client demand satisfaction constraints.

        Parameters
        ----------

        obj: T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_client : range
            The range of demand points.
        range_facility : range
            The range of facility points.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars") and hasattr(obj, "cli_assgn_vars"):
            cli_assn_vars = getattr(obj, "cli_assgn_vars")
            model = getattr(obj, "problem")
            aij = getattr(obj, "aij")

            for i in range_client:
                model += (
                    pulp.lpSum(
                        [int(aij[i, j]) * cli_assn_vars[i, j] for j in range_facility]
                    )
                    == 1
                )
        else:
            raise AttributeError(
                "Facility and client assignment variables must be set "
                "prior to adding client demand satisfaction constraints."
            )

    @staticmethod
    def add_maximal_coverage_constraint(
        obj: T_FacModel,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        Create maximal coverage constraints:

        ni_1 * y1 + ni_2 * y1 + ... + nij * yj >= xi

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_facility : range
            The range of facility points.
        range_client : range
            The range of demand points.

        Returns
        -------

        None

        """
        if hasattr(obj, "fac_vars") and hasattr(obj, "cli_vars"):
            fac_vars = getattr(obj, "fac_vars")
            dem_vars = getattr(obj, "cli_vars")
            model = getattr(obj, "problem")
            ni = getattr(obj, "aij")

            for i in range_client:
                model += (
                    pulp.lpSum([ni[i][j] * fac_vars[j] for j in range_facility])
                    >= dem_vars[i]
                )
        else:
            raise AttributeError(
                "Before setting maximal coverage constraints facility "
                "and demand variables must be set."
            )

    @staticmethod
    def add_assignment_constraint(
        obj: T_FacModel,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        Create assignment constraints.

        x_i_0 + x_i_1 + ... + x_i_j == 1

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_facility : range
            The range of facility points.
        range_client : range
            The range of demand points.

        Returns
        -------

        None

        """
        if hasattr(obj, "cli_assgn_vars"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            model = getattr(obj, "problem")

            for i in range_client:
                model += pulp.lpSum([cli_assgn_vars[i, j] for j in range_facility]) == 1
        else:
            raise AttributeError(
                "Before setting assignment constraints "
                "client assignment variables must be set."
            )

    @staticmethod
    def add_opening_constraint(
        obj: T_FacModel,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        Create opening constraints.

        y_j >= xi_j

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        range_facility : range
            The range of facility points.
        range_client : range
            The range of demand points.

        Returns
        -------

        None

        """
        if hasattr(obj, "cli_assgn_vars"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            fac_vars = getattr(obj, "fac_vars")
            model = getattr(obj, "problem")

            for i in range_client:
                for j in range_facility:
                    model += fac_vars[j] - cli_assgn_vars[i, j] >= 0
        else:
            raise AttributeError(
                "Before setting opening constraints "
                "client assignment variables must be set."
            )

    @staticmethod
    def add_minimized_maximum_constraint(
        obj: T_FacModel,
        cost_matrix: np.array,
        range_facility: range,
        range_client: range,
    ) -> None:
        """
        Create minimized maximum constraints.

        xi_0 * di_0 + xi_1 * di_1 + ... + xi_j * di_j <= W

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        cost_matrix : numpy.array
            A cost matrix in the form of a 2D array between origins and destinations.
        range_facility : range
            The range of facility points.
        range_client : range
            The range of demand points.

        Returns
        -------

        None

        Notes
        -----

        See an explanation for the ``W`` variable in
        ``spopt.locate.base.add_weight_continuous_variable``

        """
        if hasattr(obj, "cli_assgn_vars") and hasattr(obj, "weight_var"):
            cli_assgn_vars = getattr(obj, "cli_assgn_vars")
            weight_var = getattr(obj, "weight_var")
            model = getattr(obj, "problem")

            for i in range_client:
                model += (
                    pulp.lpSum(
                        [
                            cli_assgn_vars[i, j] * cost_matrix[i, j]
                            for j in range_facility
                        ]
                    )
                    <= weight_var
                )
        else:
            raise AttributeError(
                "Before setting minimized maximum constraints weight "
                "and client assignment variables must be set."
            )

    @staticmethod
    def add_p_dispersion_interfacility_constraint(
        obj: T_FacModel,
        cost_matrix: np.array,
        range_facility: range,
    ) -> None:
        """
        Create p-dispersion interfacility distance constraints.

        dij + M (2 - y_0 - y_1) >= D

        Parameters
        ----------

        obj : T_FacModel
            A bounded type of the ``LocateSolver`` class.
        cost_matrix : numpy.array
            A cost matrix in the form of a 2D array between all facility points.
        range_facility : range
            The range of facility points.

        Returns
        -------

        None

        """
        if hasattr(obj, "disperse_var") and hasattr(obj, "fac_vars"):
            M = cost_matrix.max()
            model = getattr(obj, "problem")

            for i in range_facility:
                for j in range_facility:
                    if j <= i:
                        continue
                    else:
                        dij = cost_matrix[i, j]
                        model += (
                            pulp.lpSum(
                                [(dij + M * (2 - obj.fac_vars[i] - obj.fac_vars[j]))]
                            )
                            >= obj.disperse_var
                        )
        else:
            raise AttributeError(
                "Before setting interfacility distance constraints dispersion "
                "objective value and facility assignment variables must be set."
            )

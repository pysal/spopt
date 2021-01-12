from abc import ABC, abstractmethod


class BaseSpOptSolver(ABC):
    """Base class for all spatial optimization model solvers.

    """

    @abstractmethod
    def solve(self):
        """Solve the optimization model."""
        pass


class BaseSpOptExactSolver(BaseSpOptSolver):
    """Base class for all spatial optimization model exact solvers.

    Attributes
    ----------

    spOptSolver : pywraplp.Solver
        The or-tools MIP solver.

    """

    def __init__(self, name):
        """Initialize.

        Parameters
        ----------
        name : str
            The desired name for the model.
        """
        try:
            from ortools.linear_solver import pywraplp

            self.spOptSolver = pywraplp.Solver(
                name, pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
        except ImportError:
            raise ImportError(
                "ortools is a requirement for exact solvers. "
                "you can install it with `pip install ortools`"
            )

        self.name = name

    def solve(self):
        """Solve the optimization model."""
        self.spOptSolver.Solve()


class BaseSpOptHeuristicSolver(BaseSpOptSolver):
    """Base class for all spatial optimization model heuristic solvers."""

    @abstractmethod
    def solve(self):
        """Solve the optimization model."""
        pass
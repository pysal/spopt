from abc import ABC, abstractmethod


class BaseSpOptSolver(ABC):
    """Base class for all spatial optimization model solvers."""

    @abstractmethod
    def solve(self):
        """Solve the optimization model."""
        pass


class BaseSpOptExactSolver(BaseSpOptSolver):
    """Base class for all spatial optimization model exact solvers."""

    def __init__(self, name):
        """Initialize.

        Parameters
        ----------
        name : str
            The desired name for the model.
        """
        self.name = name

    @abstractmethod
    def solve(self):
        """Solve the optimization model."""
        pass


class BaseSpOptHeuristicSolver(BaseSpOptSolver):
    """Base class for all spatial optimization model heuristic solvers."""

    @abstractmethod
    def solve(self):
        """Solve the optimization model."""
        pass

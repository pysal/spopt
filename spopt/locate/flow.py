import logging
import random
import time
import warnings
from collections import deque
from typing import Any, Literal

import numpy as np
import pandas as pd
import pulp
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from tqdm import tqdm

from .base import FacilityModelBuilder
from .util import compute_facility_usage, rising_combination


class GreedyVariable:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.varValue = value
        self.dj = 0.0


class GreedyConstraint:
    def __init__(self, name, info):
        self.name = name
        self.pi = info.get("shadow_price", 0.0)
        self.slack = info.get("slack", 0.0)


class FlowModelBuilder:
    """
    Builder class for constructing optimization variables and constraints
    for Flow Refueling Location Model (FRLM).
    """

    @staticmethod
    def add_facility_variables(
        obj: Any,
        candidate_sites: list[Any],
    ) -> None:
        """
        Add facility location binary variables.

        Parameters
        ----------
        obj : Any
            The optimization model object
        candidate_sites : list[Any]
            List of potential facility locations
        """

        fac_vars = [
            pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpInteger)
            for i in candidate_sites
        ]

        obj.facility_vars = fac_vars
        obj.fac_vars = fac_vars

    @staticmethod
    def add_ac_pc_constraints(
        obj: Any,
        a: dict[int, list[int]],
        k: dict[tuple[int, int], list[Any]],
        candidate_sites: list[Any],
    ) -> None:
        """
        The AC-PC approach decomposes each path into arcs and ensures that at least
        one facility can refuel vehicles on each arc of a path if that path is to
        be covered.

        Parameters
        ----------
        obj : Any
            The optimization model object
        a : Dict[int, list[int]]
            Arc mapping dictionary where:
            - Key: Path ID (q) - integer identifier for each OD path (1-indexed)
            - Value: List of arc IDs that compose this path

        K : Dict[Tuple[int, int], list[Any]]
            Refueling capability dictionary where:
            - Key: Tuple (path_id, arc_id) identifying a specific arc on a path
            - Value: List of candidate facility nodes that can refuel this arc
            A facility can refuel an arc if it's within vehicle_range/2 of either
            endpoint of the arc.

        candidate_sites : list[Any]
        """
        if hasattr(obj, "facility_vars") and hasattr(obj, "flow_vars"):
            facility_vars = obj.facility_vars
            flow_vars = obj.flow_vars
            model = obj.problem

            facility_index_map = {site: idx for idx, site in enumerate(candidate_sites)}

            for q in flow_vars:
                if q in a:
                    for arc in a[q]:
                        key = (q, arc)
                        if key in k:
                            refuel_facilities = pulp.lpSum(
                                facility_vars[facility_index_map[node]]
                                for node in k[key]
                                if node in facility_index_map
                            )
                            model += refuel_facilities >= flow_vars[q]

    @staticmethod
    def add_flow_variables(
        obj: Any,
        flows: dict[tuple[Any, Any], float],
        path_refueling_combinations: dict[tuple[Any, Any], list[list[Any]]]
        | None = None,
    ) -> None:
        """
        Add flow coverage continuous variables.

        Parameters
        ----------
        obj : Any
            The optimization model object
        flows : Dict[Tuple[Any, Any], float]
            Dictionary of flows between origin-destination pairs
        path_refueling_combinations : Dict[Tuple[Any, Any], list[list[Any]]], optional
            Valid facility combinations for each flow path
        """
        flow_vars = {}

        if hasattr(obj, "use_ac_pc") and obj.use_ac_pc:
            for q in range(1, len(flows) + 1):
                if q in obj.a:
                    flow_vars[q] = pulp.LpVariable(f"y_{q}", cat=pulp.LpBinary)
        elif path_refueling_combinations is None:
            for q, _od_pair in enumerate(flows.keys()):
                flow_vars[q] = pulp.LpVariable(
                    f"y_{q}", lowBound=0, upBound=1, cat=pulp.LpContinuous
                )
        else:
            for q, od_pair in enumerate(flows.keys()):
                valid_combinations = path_refueling_combinations[od_pair]
                for h, _combination in enumerate(valid_combinations):
                    flow_vars[(q, h)] = pulp.LpVariable(
                        f"y_{q}_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )

        obj.flow_vars = flow_vars

    @staticmethod
    def add_flow_coverage_constraints(
        obj: Any,
        flows: dict[tuple[Any, Any], float],
        path_refueling_combinations: dict[tuple[Any, Any], list[list[Any]]]
        | None = None,
    ) -> None:
        """
        Add flow coverage constraints.

        Parameters
        ----------
        obj : Any
            The optimization model object
        flows : Dict[Tuple[Any, Any], float]
            Dictionary of flows between origin-destination pairs
        path_refueling_combinations :Dict[Tuple[Any, Any], list[list[Any]]], optional
            Valid facility combinations for each flow path
        """
        if hasattr(obj, "flow_vars") and hasattr(obj, "facility_vars"):
            flow_vars = obj.flow_vars
            model = obj.problem

            if path_refueling_combinations is None:
                for q, _od_pair in enumerate(flows.keys()):
                    model += flow_vars[q] <= 1
            else:
                for q, od_pair in enumerate(flows.keys()):
                    model += (
                        pulp.lpSum(
                            flow_vars[(q, h)]
                            for h in range(len(path_refueling_combinations[od_pair]))
                        )
                        <= 1
                    )
        else:
            raise AttributeError(
                "Flow and facility variables must be set before adding flow "
                "coverage constraints."
            )

    @staticmethod
    def add_capacity_constraints(
        obj: Any,
        flows: dict[tuple[Any, Any], float],
        path_refueling_combinations: dict[tuple[Any, Any], list[list[Any]]],
        e_coefficients: dict[tuple[Any, Any], float],
        capacity: float,
        candidate_sites: list[Any],
    ) -> None:
        """
        Add facility capacity constraints.

        Parameters
        ----------
        obj : Any
            The optimization model object
        flows : Dict[Tuple[Any, Any], float]
            Dictionary of flows between origin-destination pairs
        path_refueling_combinations : Dict[Tuple[Any, Any], list[list[Any]]]
            Valid facility combinations for each flow path
        e_coefficients : Dict[Tuple[Any, Any], float]
            Refueling frequency coefficients
        capacity : float
            Facility module capacity
        candidate_sites : list[Any]
            List of potential facility locations
        """
        if hasattr(obj, "facility_vars") and hasattr(obj, "flow_vars"):
            facility_vars = obj.facility_vars
            flow_vars = obj.flow_vars
            model = obj.problem

            for k in candidate_sites:
                model += (
                    pulp.lpSum(
                        e_coefficients[od_pair]
                        * compute_facility_usage(od_pair[0], od_pair[1], k, combination)
                        * flows[od_pair]
                        * flow_vars[(q, h)]
                        for q, od_pair in enumerate(flows.keys())
                        for h, combination in enumerate(
                            path_refueling_combinations[od_pair]
                        )
                        if k in combination
                    )
                    <= capacity * facility_vars[k]
                )
        else:
            raise AttributeError(
                "Facility and flow variables must be set before adding capacity "
                "constraints."
            )

    @staticmethod
    def add_threshold_constraints(
        obj: Any,
        flows: dict[tuple[Any, Any], float],
        node_weights: np.ndarray,
        threshold: float,
        weight: float,
    ) -> None:
        """
        Add threshold extension constraints to the model.

        Parameters
        ----------
        obj : Any
            The optimization model object
        flows : Dict[Tuple[Any, Any], float]
            Dictionary of flows between origin-destination pairs
        node_weights : np.ndarray
            Weights for each node
        threshold : float
            Minimum flow coverage threshold
        weight : float
            Weight parameter for combining node and flow coverage
        """
        if hasattr(obj, "facility_vars") and hasattr(obj, "flow_vars"):
            model = obj.problem
            flow_vars = obj.flow_vars

            node_coverage_vars = {}
            for origin in {od[0] for od in flows}:
                node_coverage_vars[origin] = pulp.LpVariable(
                    f"node_coverage_{origin}", lowBound=0, upBound=1, cat=pulp.LpBinary
                )

            for origin in node_coverage_vars:
                origin_flows = [
                    (q, od_pair)
                    for q, od_pair in enumerate(flows.keys())
                    if od_pair[0] == origin
                ]

                total_origin_flow = sum(flows[od_pair] for _, od_pair in origin_flows)

                model += (
                    pulp.lpSum(
                        flows[od_pair] * flow_vars[(q, h)]
                        for q, od_pair in origin_flows
                        for h in range(len(obj.path_refueling_combinations[od_pair]))
                    )
                    >= threshold * total_origin_flow * node_coverage_vars[origin]
                )

            if hasattr(obj, "original_objective"):
                original_objective = obj.original_objective

                node_coverage_term = pulp.lpSum(
                    node_weights[origin] * node_coverage_vars[origin]
                    for origin in node_coverage_vars
                )

                flow_coverage_term = original_objective

                model += weight * node_coverage_term + (1 - weight) * flow_coverage_term

            obj.node_coverage_vars = node_coverage_vars
        else:
            raise AttributeError(
                "Facility and flow variables must be set before adding threshold "
                "constraints."
            )

    @staticmethod
    def add_combination_variables(
        obj: Any,
    ) -> None:
        """
        Add combination variables vh for basic FRLM.
        """
        combination_vars = {}

        # Create a mapping of combinations
        h = 0
        combination_mapping = {}

        for _od_pair, combinations in obj.path_refueling_combinations.items():
            for combo in combinations:
                # Convert list to tuple for hashing
                combo_tuple = tuple(sorted(combo))
                if combo_tuple not in combination_mapping:
                    combination_mapping[combo_tuple] = h
                    combination_vars[h] = pulp.LpVariable(
                        f"v_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )
                    h += 1

        obj.combination_vars = combination_vars
        obj.combination_mapping = combination_mapping

    @staticmethod
    def add_combination_refueling_constraints(
        obj: Any, flows: dict[tuple[Any, Any], float]
    ) -> None:
        """
        Add constraints linking combinations to flow coverage.
        """
        model = obj.problem
        flow_vars = obj.flow_vars
        combination_vars = obj.combination_vars
        combination_mapping = obj.combination_mapping
        facility_vars = obj.facility_vars

        for q, od_pair in enumerate(flows.keys()):
            if od_pair in obj.path_refueling_combinations:
                valid_combos = []
                for combo in obj.path_refueling_combinations[od_pair]:
                    combo_tuple = tuple(sorted(combo))
                    h = combination_mapping[combo_tuple]
                    valid_combos.append(combination_vars[h])

                if valid_combos:
                    model += flow_vars[q] <= pulp.lpSum(valid_combos)

        for combo_tuple, h in combination_mapping.items():
            for facility in combo_tuple:
                facility_idx = obj.candidate_sites.index(facility)
                model += combination_vars[h] <= facility_vars[facility_idx]


class FRLMCoverageMixin:
    """Mixin to calculate flow coverage statistics for FRLM."""

    def get_flow_coverage(self) -> None:
        """Improved and fixed calculation of flow coverage."""
        if not hasattr(self, "facility_vars") or self.facility_vars is None:
            raise AttributeError(
                "Model must be solved before calculating coverage. Call solve() first."
            )

        if not self.flow_coverage:
            self.flow_coverage = {}

            if self.use_ac_pc:
                flow_list = list(self.flows.items())
                for q in range(1, len(flow_list) + 1):
                    if q in self.flow_vars and self.flow_vars[q].varValue is not None:
                        od_pair, flow_volume = flow_list[q - 1]
                        covered_proportion = self.flow_vars[q].varValue

                        self.flow_coverage[od_pair] = {
                            "flow_volume": flow_volume,
                            "covered_proportion": covered_proportion,
                            "covered_volume": flow_volume * covered_proportion,
                        }
            else:
                for q, od_pair in enumerate(self.flows.keys()):
                    flow_volume = self.flows[od_pair]
                    covered_proportion = 0.0

                    if od_pair in self.path_refueling_combinations:
                        for h in range(len(self.path_refueling_combinations[od_pair])):
                            if (q, h) in self.flow_vars and self.flow_vars[
                                (q, h)
                            ].varValue is not None:
                                covered_proportion += self.flow_vars[(q, h)].varValue

                        self.flow_coverage[od_pair] = {
                            "flow_volume": flow_volume,
                            "covered_proportion": round(covered_proportion, 2),
                            "covered_volume": round(
                                flow_volume * covered_proportion, 2
                            ),
                        }

        total_covered_volume = sum(
            coverage["covered_volume"] for coverage in self.flow_coverage.values()
        )
        flow_volume = sum(
            coverage["flow_volume"] for coverage in self.flow_coverage.values()
        )
        return {
            "covered_volume": round(total_covered_volume, 2),
            "flow_volume": round(flow_volume, 2),
            "covered_proportion": round(total_covered_volume / flow_volume, 2),
        }

    def get_vmt_coverage(self) -> dict[str, float]:
        if not hasattr(self, "_path_distances"):
            self._calculate_path_distances()

        if not hasattr(self, "flow_coverage"):
            self.calculate_flow_coverage()

        total_vmt = 0
        covered_vmt = 0

        for od_pair, coverage in self.flow_coverage.items():
            path_vmt = self.flows[od_pair] * self._path_distances.get(od_pair, 0)
            total_vmt += path_vmt
            covered_vmt += path_vmt * coverage.get("covered_proportion", 0)

        return {
            "total_vmt": total_vmt,
            "covered_vmt": covered_vmt,
            "vmt_coverage_percentage": covered_vmt / total_vmt if total_vmt > 0 else 0,
            "vmt_details": self.flow_coverage.copy(),
        }


class FRLMNodeCoverageMixin:
    def calculate_covered_nodes(self) -> None:
        if self.threshold <= 0:
            self.covered_nodes = []
            return

        if not hasattr(self, "flow_coverage"):
            raise AttributeError(
                "Flow coverage must be calculated first. Call get_flow_coverage()."
            )

        covered_nodes = []

        origin_flows = {}
        for od_pair, flow_volume in self.flows.items():
            origin = od_pair[0]
            if origin not in origin_flows:
                origin_flows[origin] = []
            origin_flows[origin].append((od_pair, flow_volume))

        for origin, flows in origin_flows.items():
            total_origin_flow = sum(volume for _, volume in flows)
            covered_origin_flow = sum(
                volume
                for od_pair, volume in flows
                if self.flow_coverage.get(od_pair, {}).get("covered_proportion", 0) > 0
            )

            if covered_origin_flow >= self.threshold * total_origin_flow:
                covered_nodes.append(origin)

        self.covered_nodes = covered_nodes

    def get_node_coverage_percentage(self) -> float:
        """Get percentage of nodes covered."""
        if not hasattr(self, "covered_nodes"):
            self.calculate_covered_nodes()

        total_origins = len({od[0] for od in self.flows})
        return len(self.covered_nodes) / total_origins if total_origins > 0 else 0.0


class FRLMSolverStatsMixin:
    def extract_solver_statistics(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise AttributeError("Model must be solved first. Call solve().")

        self.solver_stats = {
            "solver_name": self.solver_type,
            "status": self.status,
            "solution_time": self.solution_time,
            "objective_value": self.objective_value,
        }

        if self.solver_type == "pulp":
            self.solver_stats.update(
                {
                    "num_variables": len(self.model.variables()),
                    "num_constraints": len(self.model.constraints),
                    "pulp_status": self.pulp_status,
                }
            )

            if hasattr(self.model, "constraints"):
                self.shadow_prices = {}
                for name, constraint in self.model.constraints.items():
                    if hasattr(constraint, "pi") and constraint.pi is not None:
                        self.shadow_prices[name] = constraint.pi

            if hasattr(self.model, "variables"):
                self.reduced_costs = {}
                for var in self.model.variables():
                    if hasattr(var, "dj") and var.dj is not None:
                        self.reduced_costs[var.name] = var.dj

        elif self.solver_type == "greedy":
            self.solver_stats.update(
                {
                    "iterations_completed": len(self.greedy_iterations),
                    "marginal_contributions": self.marginal_contributions,
                }
            )

    def get_detailed_results(self) -> dict:
        if not hasattr(self, "solver_stats"):
            self.extract_solver_statistics()

        model_params = {
            "vehicle_range": round(self.vehicle_range, 2),
            "p_facilities": self.p_facilities,
            "capacity": self.capacity,
            "threshold": self.threshold,
            "objective_type": self.objective,
        }

        if self.threshold > 0:
            model_params["weight"] = self.weight

        results = {
            "model_parameters": model_params,
            "solution": {
                "status": self.status,
                "objective_value": self.objective_value,
                "selected_facilities": list(self.selected_facilities.keys()),
                "solution_time": self.solution_time,
            },
            "solver_statistics": self.solver_stats,
        }

        if hasattr(self, "flow_coverage"):
            results["coverage_statistics"] = {
                "flow_breakdown": self.flow_coverage,
            }

            if self.objective == "vmt":
                results["coverage_statistics"].update(self.get_vmt_coverage())

        if self.threshold > 0 and hasattr(self, "covered_nodes"):
            results["node_coverage"] = {
                "covered_nodes": self.covered_nodes,
                "node_coverage_percentage": self.get_node_coverage_percentage(),
            }

        return results


class FRLM(FRLMCoverageMixin, FRLMNodeCoverageMixin, FRLMSolverStatsMixin):
    """
    Flow Refueling Location Model (FRLM)

    Parameters
    ----------
    vehicle_range : Union[float, int], default=200000.0
        Defines the maximum travel distance or range for vehicles.
        - If value is between 0 and 1: Treated as a percentage of the longest path.
          e.g. 0.5: 50% of the longest path in the network
        - If value > 1: Treated as an absolute distance in distance units (e.g., meters)

    p_facilities : int, default=5
        Number of facility modules to be located in the network.

    capacity : Union[float, None], default=None
        Facility module capacity constraint.
        - If None: Uses an uncapacitated model
        - If provided: Sets a maximum capacity for each facility module

    threshold : float, default=0.0
        Minimum flow coverage percentage for a node to be considered covered,
        0 disables threshold extension.

    weight : float, default=0.99
        Controls the trade-off between node coverage and flow coverage for
        threshold extension objective.
        Range: [0.0, 1.0]
        - 1.0: Prioritises node coverage
        - 0.0: Prioritises flow coverage
        Only used when threshold > 0.

    objective : str, default="flow"
        Optimisation objective.
        - "flow": Maximize total flow coverage
        - "vmt": Maximize vehicle miles traveled (VMT) coverage

    include_destination : bool, default=False
        Determines how node weights are calculated in threshold extension.

        - False: Only origin nodes contribute to weight calculation
        - True: Both origin and destination nodes contribute to weight calculation
    """

    def __init__(
        self,
        vehicle_range: float | int = 200000,
        p_facilities: int = 5,
        capacity: float | None = None,
        threshold: float = 0.0,
        weight: float = 0.99,
        objective: str = "flow",
        include_destination: bool = False,
    ):
        if vehicle_range <= 0:
            raise ValueError("vehicle_range must be positive")

        self.vehicle_range = vehicle_range
        self._original_vehicle_range = vehicle_range
        self.p_facilities = p_facilities

        self.capacity = capacity

        self.k = {}
        self.a = {}
        self.use_ac_pc = False

        self.threshold = threshold
        self.weight = weight

        self.objective = objective
        self.include_destination = include_destination

        self.network = None
        self.flows = {}
        self.flow_paths = {}
        self.candidate_sites = []
        self.facility_combinations = []
        self.path_refueling_combinations = {}
        self.e_coefficients = {}
        self.scipy_network = None
        self.node_weights = None

        self.flow_coverage = {}
        self.covered_nodes = []
        self.objective_value = 0
        self.solution_time = 0
        self.status = None

        self.solver_type = None
        self.variables = {}
        self.constraints = {}

        self.model = pulp.LpProblem(f"FRLM_{self.objective.upper()}", pulp.LpMaximize)
        self.facility_vars = None
        self.flow_vars = None
        self.node_coverage_vars = None
        self._model_built = False

        self.lagrange_multipliers = {}
        self.reduced_costs = {}
        self.shadow_prices = {}

        self.greedy_iterations = []
        self.marginal_contributions = {}

        self.solver_stats = {}

    def _initialize_variables(self):
        self.variables = {}
        self.constraints = {}
        self.lagrange_multipliers = {}
        self.reduced_costs = {}
        self.shadow_prices = {}
        self.greedy_iterations = []
        self.marginal_contributions = {}
        self.solver_stats = {}
        self.pulp_status = None

    def get_constraint_dual(self, constraint_name: str) -> float | None:
        return self.lagrange_multipliers.get(constraint_name)

    @classmethod
    def from_flow_dataframe(
        cls, network: sp.csr_matrix, flows: pd.DataFrame | dict, **kwargs
    ):
        """
        Create FRLM instance from network and flows.

        Parameters
        ----------
        network : sp.csr_matrix
            Network as a scipy sparse matrix
        flows : Union[pd.DataFrame, Dict]
            Flow data
        """
        frlm = cls(**kwargs)

        frlm.scipy_network = network
        frlm.network = network
        frlm.candidate_sites = list(range(network.shape[0]))

        if flows is not None:
            if isinstance(flows, dict):
                frlm._load_flows_from_dict(flows)
            elif isinstance(flows, pd.DataFrame):
                required_columns = ["origin", "destination", "volume"]
                if not all(col in flows.columns for col in required_columns):
                    raise ValueError(
                        f"Flows DataFrame must contain columns: {required_columns}"
                    )
                path_col = "path" if "path" in flows.columns else None
                frlm._load_flows_from_dataframe(flows, path_col=path_col)
            else:
                raise TypeError("Flows must be a dictionary or DataFrame")
        else:
            warnings.warn(
                "No flows provided. FRLM will not have any flow data to work with.",
                UserWarning,
                stacklevel=2,
            )

        frlm._build_model()

        return frlm

    def add_network(self, network: sp.csr_matrix):
        """
        Add network to the FRLM instance.

        Parameters
        ----------
        network : scipy.sparse.csr_matrix
            The network as a scipy sparse matrix where element [i,j] represents
            the distance/cost between nodes i and j.

        Returns
        -------
        self : FRLM
            Returns the instance for method chaining
        """
        if not isinstance(network, sp.csr_matrix):
            raise TypeError("Network must be a scipy.sparse.csr_matrix")

        self.scipy_network = network
        self.network = network
        self.candidate_sites = list(range(network.shape[0]))
        return self

    def add_flows(self, flows: pd.DataFrame | dict[tuple[Any, Any], float]):
        """
        Add multiple flows to the FRLM instance.

        Parameters
        ----------
        flows : Union[pd.DataFrame, Dict[Tuple[Any, Any], float]]
            Flows to be added. Can be either:
            - A pandas DataFrame with flow information
            - A dictionary mapping (origin, destination) tuples to flow volumes

        Returns
        -------
        self : FRLM
            Returns the instance for method chaining
        """
        if self.scipy_network is None:
            raise ValueError("Network must be added before adding flows")

        if isinstance(flows, dict):
            self._load_flows_from_dict(flows)
        elif isinstance(flows, pd.DataFrame):
            self._load_flows_from_dataframe(flows)
        else:
            raise TypeError("Flows must be a dictionary or pandas.DataFrame")

        return self

    def _reconstruct_path(
        self, predecessors: list[int], origin: int, destination: int
    ) -> list[int]:
        """
        Reconstruct the shortest path between origin and destination.

        Parameters
        ----------
        predecessors : list[int]
            Predecessor array from shortest path algorithm
        origin : int
            Starting node of the path
        destination : int
            Ending node of the path

        Returns
        -------
        list[int]
            Shortest path from origin to destination
        """
        if predecessors[destination] == -9999:
            raise KeyError(f"No path exists between {origin} and {destination}")

        path = deque([destination])
        current = destination

        while current != origin:
            prev = predecessors[current]
            if prev == -9999:
                raise KeyError(f"No path exists between {origin} and {destination}")

            path.appendleft(prev)
            current = prev

        if path[0] != origin:
            path.appendleft(origin)

        return list(path)

    def add_flow(
        self, origin: Any, destination: Any, volume: float, path: list | None = None
    ) -> None:
        """
        Add flow with path calculation using scipy network.

        Parameters
        ----------
        origin : Any
            Origin node ID
        destination : Any
            Destination node ID
        volume : float
            Flow volume
        path : Optional[List], default=None
            Specific path to use (if None, shortest path is computed)
        """

        num_nodes = self.scipy_network.shape[0]
        if (
            origin < 0
            or origin >= num_nodes
            or destination < 0
            or destination >= num_nodes
        ):
            raise KeyError(
                f"Origin {origin} or destination {destination} is not in the network"
            )

        self.flows[(origin, destination)] = volume

        if path is not None:
            self.flow_paths[(origin, destination)] = path
            self._set_adaptive_vehicle_range()
            return

        if (origin, destination) not in self.flow_paths:
            try:
                distances, predecessors = csgraph.shortest_path(
                    self.scipy_network,
                    directed=False,
                    indices=origin,
                    return_predecessors=True,
                )

                if np.isinf(distances[destination]):
                    raise KeyError(f"No path exists between {origin} and {destination}")

                path = self._reconstruct_path(predecessors, origin, destination)
                if path[0] != origin or path[-1] != destination:
                    raise KeyError(
                        f"Path reconstruction failed for {origin} to {destination}"
                    )

            except Exception as e:
                print(f"{e}")
                raise ValueError(
                    f"Error finding path between {origin} and {destination}: {e}"
                ) from e

            self.flow_paths[(origin, destination)] = path

        self._set_adaptive_vehicle_range()

    def _load_flows_from_dict(self, flows: dict[tuple[Any, Any], float]) -> None:
        """
        Load flows from dictionary by converting to DataFrame and using
        _load_flows_from_dataframe().

        Parameters
        ----------
        flows : Dict[Tuple[Any, Any], float]
            Dictionary mapping (origin, destination) tuples to flow volumes
        """
        data = []
        for (origin, destination), volume in flows.items():
            data.append(
                {"origin": origin, "destination": destination, "volume": volume}
            )

        df = pd.DataFrame(data)

        self._load_flows_from_dataframe(
            df,
            origin_col="origin",
            destination_col="destination",
            volume_col="volume",
            path_col=None,
        )

    def _load_flows_from_dataframe(
        self,
        df: pd.DataFrame,
        origin_col: str = "origin",
        destination_col: str = "destination",
        volume_col: str = "volume",
        path_col: str | None = None,
    ) -> None:
        """
        origin_col : str, default='origin'
            Column name for origin nodes
        destination_col : str, default='destination'
            Column name for destination nodes
        volume_col : str, default='volume'
            Column name for flow volumes
        path_col : Optional[str], default=None
            Column name for pre-computed paths.
            Expected to be a list of integer node IDs.

        Raises
            TypeError
                If the path is not a list when provided.
            ValueError
                If the path is invalid or inconsistent with origin and destination.
        """
        for _, row in df.iterrows():
            origin = row[origin_col]
            destination = row[destination_col]
            volume = row[volume_col]

            path = None
            if path_col is not None and path_col in df.columns:
                path = row[path_col]

                if not isinstance(path, list):
                    raise TypeError("Path must be a list of integer node ids. ")

                if path and (path[0] != origin or path[-1] != destination):
                    raise ValueError(
                        f"Path for OD pair ({origin}, {destination}) is "
                        f"inconsistent. Path should start at {origin} and end at "
                        f"{destination}."
                    )

            self.add_flow(origin, destination, volume, path)

        self._set_adaptive_vehicle_range()

    def _set_adaptive_vehicle_range(self):
        """Set vehicle range based on percentage of longest path."""
        if not self.flow_paths or not (0 < self._original_vehicle_range <= 1):
            return

        scipy_network = self.scipy_network.tocsr()

        max_distance = 0

        for path in self.flow_paths.values():
            path_distance = 0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                try:
                    path_distance += scipy_network[u, v]

                except Exception as e:
                    print(f"Distance calculation error: {e}")
                    path_distance += 1.0

            max_distance = max(max_distance, path_distance)

        self.vehicle_range = max_distance * self._original_vehicle_range

    def compute_refueling_frequency(self, origin: Any, destination: Any) -> float:
        od_pair = (origin, destination)

        if od_pair not in self.flow_paths:
            raise KeyError(f"No path exists for OD pair {od_pair}")

        path = self.flow_paths[od_pair]

        scipy_network = self.scipy_network.tocsr()

        roundtrip_distance = 0
        for i in range(len(path) - 1):
            try:
                roundtrip_distance += scipy_network[path[i], path[i + 1]]
            except TypeError:
                roundtrip_distance += 1.0

        roundtrip_distance *= 2

        if roundtrip_distance <= 0:
            return 1.0

        trips_per_tank = max(1, int(self.vehicle_range / roundtrip_distance))
        e_q = 1.0 / trips_per_tank

        self.e_coefficients[od_pair] = e_q
        return e_q

    def check_path_refueling_feasibility(self, path: list, facilities: list) -> bool:
        """
        Check if a path can be traversed with given facilities using scipy network.

        Parameters
        ----------
        path : List
            Nodes in the path
        facilities : List
            Potential refueling facilities

        Returns
        -------
        bool
            Whether the path is feasible with given facilities
        """
        if len(path) < 2:
            return False

        distances = csgraph.shortest_path(self.scipy_network, method="auto")

        total_length = sum(
            distances[path[i], path[i + 1]] for i in range(len(path) - 1)
        )

        if total_length * 2 <= self.vehicle_range:
            return any(facility in path for facility in facilities)

        remaining_range = self.vehicle_range / 2
        if path[0] in facilities:
            remaining_range = self.vehicle_range

        for i in range(1, len(path)):
            segment_length = distances[path[i - 1], path[i]]
            remaining_range -= segment_length

            if remaining_range < 0:
                return False

            if path[i] in facilities:
                remaining_range = self.vehicle_range

        remaining_range = self.vehicle_range / 2
        if path[-1] in facilities:
            remaining_range = self.vehicle_range

        for i in range(len(path) - 1, 0, -1):
            segment_length = distances[path[i - 1], path[i]]
            remaining_range -= segment_length

            if remaining_range < 0:
                return False

            if path[i - 1] in facilities:
                remaining_range = self.vehicle_range

        return True

    def generate_path_refueling_combinations(
        self,
        facility_combinations: list[list] | None = None,
        start: int | None = None,
        stop: int | None = None,
        method: str = "auto",
    ) -> dict:
        """
        Generate dictionary mapping OD pairs to valid facility combinations

        Parameters
        ----------
        facility_combinations : Optional[List[List]], optional
            Pre-computed facility combinations (only used for combination method)
        start : int, optional
            Minimum size of combinations (for combination method)
        stop : int, optional
            Maximum size of combinations (for combination method)
        method : str, optional
            Method to use: "auto", "combination", or "ac_pc"
            - "auto": Automatically choose based on model type
            - "combination": Generate all possible facility combinations
            - "ac_pc": Generate K and a sets for Arc Cover Path Cover

        Returns
        -------
        Dict
            Dictionary of OD pairs to valid facility combinations (combination method)
            or sets K and a sets (ac_pc method)
        """

        method = "combination" if self.capacity is not None else "ac_pc"

        if method == "ac_pc":
            if self.scipy_network is None:
                raise ValueError("No network loaded")

            distances = csgraph.shortest_path(self.scipy_network, method="auto")

            self.k = {}
            self.a = {}

            for path_id, (_od_pair, path) in enumerate(self.flow_paths.items(), 1):
                arc_ids = []

                for arc_id, i in enumerate(range(len(path) - 1), 1):
                    u, v = path[i], path[i + 1]
                    arc_ids.append(arc_id)

                    refueling_nodes = []
                    for candidate in self.candidate_sites:
                        dist_to_u = distances[candidate, u]
                        dist_to_v = distances[candidate, v]

                        if (
                            dist_to_u <= self.vehicle_range / 2
                            or dist_to_v <= self.vehicle_range / 2
                        ):
                            refueling_nodes.append(candidate)

                    self.k[(path_id, arc_id)] = refueling_nodes

                self.a[path_id] = arc_ids

            for od_pair in self.flow_paths:
                self.compute_refueling_frequency(od_pair[0], od_pair[1])

            self.use_ac_pc = True

            return {"k": self.k, "a": self.a}

        else:
            # combination method for capacitated models
            self.use_ac_pc = False

            if facility_combinations is not None:
                self.facility_combinations = facility_combinations
            else:
                start = start if start is not None else 1
                stop = stop if stop is not None else self.p_facilities

                self.facility_combinations = list(
                    rising_combination(self.candidate_sites, start=start, stop=stop)
                )

            path_refueling_combinations = {}

            for od_pair, path in tqdm(
                self.flow_paths.items(), desc="Generating combinations"
            ):
                valid_combinations = []

                for combo in self.facility_combinations:
                    if self.check_path_refueling_feasibility(path, combo):
                        valid_combinations.append(combo)

                path_refueling_combinations[od_pair] = valid_combinations

                self.compute_refueling_frequency(od_pair[0], od_pair[1])

            self.path_refueling_combinations = path_refueling_combinations

            return path_refueling_combinations

    def calculate_node_weights(self, include_destination: bool = False) -> np.ndarray:
        """
           Calculate node weights based on flow volumes.

        Parameters
        ----------
        include_destination : bool, default=False
            If True, include destination nodes in weight calculation.
            If False, only use origin nodes for weight calculation, as specified
            in the paper (Hong and Kuby, 2016).

        Returns
        -------
        np.ndarray
            Array of node weights
        """
        if not self.flows:
            raise ValueError("Flows must be loaded first")

        max_node = max(max(od[0], od[1]) for od in self.flows)
        node_weights = np.zeros(max_node + 1)
        total_flow = sum(self.flows.values())

        for (origin, destination), volume in self.flows.items():
            node_weights[origin] += volume

            if include_destination:
                node_weights[destination] += volume

        node_weights = node_weights / (total_flow * (2 if include_destination else 1))

        self.node_weights = node_weights
        return node_weights

    def _calculate_objective_value(
        self, flow_coverage: dict, objective: str = "flow"
    ) -> float:
        """
        Calculate objective value based on coverage and objective type.

        Parameters
        ----------
        flow_coverage : Dict
            Dictionary with OD pairs as keys and coverage info as values
        objective : str
            "flow" or "vmt"

        Returns
        -------
        float
            Objective value
        """
        if objective == "vmt":
            if not hasattr(self, "_path_distances"):
                self._path_distances = {}
                scipy_network = self.scipy_network.tocsr()
                for od_pair, path in self.flow_paths.items():
                    distance = 0
                    for i in range(len(path) - 1):
                        try:
                            distance += scipy_network[path[i], path[i + 1]]
                        except (KeyError, IndexError):
                            distance += 1.0
                    self._path_distances[od_pair] = distance

            total_vmt = 0
            for od_pair, coverage in flow_coverage.items():
                vmt = (
                    coverage.get("covered_proportion", 0.0)
                    * self.flows[od_pair]
                    * self._path_distances.get(od_pair, 0)
                )
                total_vmt += vmt

            return total_vmt
        else:
            total_covered_flow = 0
            for _od_pair, coverage in flow_coverage.items():
                covered_flow = coverage.get("covered_volume", 0.0)
                total_covered_flow += covered_flow

            return total_covered_flow

    def _build_model(
        self, objective: str = None, facility_combinations: list[list] | None = None
    ) -> None:
        """
        Build the optimization model with variables and constraints.

        Parameters
        ----------
        objective : str, optional
            Override the instance's objective type
        facility_combinations : Optional[list[list]], optional
            Pre-computed facility combinations
        """
        if self.scipy_network is None:
            raise ValueError("No network loaded.")

        if not self.flows:
            raise ValueError("No flows loaded.")

        objective = objective or self.objective

        if self.capacity is not None:
            if (
                not hasattr(self, "path_refueling_combinations")
                or not self.path_refueling_combinations
            ):
                self.generate_path_refueling_combinations(
                    method="combination", facility_combinations=facility_combinations
                )
        else:
            if not hasattr(self, "k") or not self.k:
                self.generate_path_refueling_combinations(method="ac_pc")

        self._build_pulp_model(
            objective=objective,
            facility_combinations=facility_combinations,
            threshold=self.threshold,
            weight=self.weight,
            include_destination=self.include_destination,
        )

        self._model_built = True

    def solve(
        self,
        solver: Literal["greedy"] | pulp.LpSolver = None,
        seed: int | None = None,
        initialization_method: str = "empty",
        max_iterations: int = 100,
        **kwargs,
    ) -> dict:
        """
        Solve the FRLM problem.

        Parameters
        ----------
        solver : Union[Literal["greedy"], pulp.LpSolver]
            Solver to use. If None, PuLP will use its default solver.
            If "greedy", uses greedy heuristic.
            If a pulp.Solver instance, uses that solver.
        seed : Optional[int], default=None
            Random seed for reproducibility
        initialization_method : str, default="empty"
            Method for initializing greedy solution
        max_iterations : int, default=100
            Maximum iterations for greedy solver
        **kwargs
            Additional solver-specific parameters
        """
        if not self._model_built:
            self._build_model()

        threshold = kwargs.get("threshold", self.threshold)
        weight = kwargs.get("weight", self.weight)

        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        if weight < 0 or weight > 1:
            raise ValueError(f"Weight must be between 0 and 1, got {weight}")

        self._initialize_variables()

        if solver == "greedy":
            self.solver_type = "greedy"
            self._solve_greedy(
                objective=kwargs.get("objective", self.objective),
                seed=seed,
                initialization_method=initialization_method,
                max_iterations=max_iterations,
                **kwargs,
            )

            if self.threshold > 0:
                self.calculate_covered_nodes()
            self._update_pulp_variables_from_solution()
            self.model.assignStatus(pulp.LpStatusOptimal)
            self.pulp_status = pulp.LpStatusOptimal

        else:
            self.solver_type = "pulp"
            self._solve_pulp(
                solver_instance=solver,
                objective=kwargs.get("objective", self.objective),
                **kwargs,
            )

        if self.status in ["Optimal", "Heuristic"]:
            self.get_flow_coverage()

        return {
            "status": self.status,
            "objective_value": round(self.objective_value),
            "selected_facilities": list(self.selected_facilities.keys()),
        }

    def _solve_greedy(
        self,
        objective: str = "flow",
        seed: int | None = None,
        initialization_method: str = "empty",
        max_iterations: int = 100
    ) -> dict:
        """
        Solve using greedy heuristic for facility location.

        Greedy Solver :
        1. Initialisation Phase:
            - Start with an empty set of facilities or a small initial set
            - Goal: Incrementally build a solution by adding facilities
            - Each iteration aims to maximise the objective function

        2. Objective Function Evaluation:
            At each iteration, evaluate each candidate facility by
                tentatively adding it to the current solution.
            Measure how much it improves the objective:
                - For flow: sum of flow volumes covered.
                - For vmt: flow volume × path distance.
                - If a threshold is set: consider both node coverage and flow coverage
                using a weighted combination.

        3. Selection Criteria:
            - Choose the facility that provides the maximum marginal benefit
            - Update the current solution by adding that facility
            - Iteratively select facilities until:
                a) Desired number of facilities is reached
                b) No further improvement is possible
                c) Maximum iterations are reached
        """

        self.solver_type = "greedy"

        if seed is not None:
            np.random.seed(seed)
            logging.log(logging.INFO, f"Random seed set to: {seed}")

        start_time = time.time()

        if objective == "vmt" and not hasattr(self, "_path_distances"):
            self._calculate_path_distances()

        current_facilities = {}
        current_objective = 0
        iterations = []
        marginal_contributions = {}

        initial_facilities = self._initialize_greedy_solution(initialization_method)
        for facility in initial_facilities:
            current_facilities[facility] = 1

        if initial_facilities:
            evaluation = self._evaluate_solution_with_combinations(current_facilities)
            current_objective = self._calculate_objective_value(
                evaluation["flow_coverage"], objective
            )

        remaining_facilities = self.p_facilities - sum(current_facilities.values())

        for iteration in range(remaining_facilities):
            if iteration >= max_iterations:
                break

            best_location = None
            best_improvement = 0
            iteration_candidates = {}

            for candidate_facility in self.candidate_sites:
                candidate_facilities = current_facilities.copy()

                if self.capacity is not None:
                    # Capacitated: can add multiple modules
                    candidate_facilities[candidate_facility] = (
                        candidate_facilities.get(candidate_facility, 0) + 1
                    )
                else:
                    # Uncapacitated: skip if already selected
                    if candidate_facility in current_facilities:
                        continue
                    candidate_facilities[candidate_facility] = 1

                # Evaluate candidate solution
                evaluation = self._evaluate_solution_with_combinations(
                    candidate_facilities
                )
                self.flow_coverage = evaluation["flow_coverage"]
                candidate_obj_value = self._calculate_objective_value(
                    evaluation["flow_coverage"], objective
                )
                improvement = candidate_obj_value - current_objective

                # Track candidate information
                iteration_candidates[candidate_facility] = {
                    "facility_id": candidate_facility,
                    "objective_value": candidate_obj_value,
                    "marginal_benefit": improvement,
                }

                # Update best if improvement is better
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_location = candidate_facility

            # Record iteration details
            iteration_info = {
                "iteration": sum(current_facilities.values()) + iteration + 1,
                "selected_facility": best_location,
                "objective_value": current_objective + best_improvement,
                "candidates_evaluated": iteration_candidates,
                "marginal_benefit": best_improvement,
            }
            iterations.append(iteration_info)

            # Stop if no improvement found
            if best_location is None or best_improvement <= 0:
                break

            # Update current solution
            if self.capacity is not None:
                current_facilities[best_location] = (
                    current_facilities.get(best_location, 0) + 1
                )
            else:
                current_facilities[best_location] = 1

            current_objective += best_improvement
            marginal_contributions[best_location] = best_improvement

        self.objective_value = current_objective
        # Final evaluation
        self.solution_time = round(time.time() - start_time, 2)
        self.status = "Heuristic"
        for k, site in enumerate(self.candidate_sites):
            self.facility_vars[k].varValue = current_facilities.get(site, 0)
        self.greedy_iterations = iterations
        self.marginal_contributions = marginal_contributions
        result = {
            "status": self.status,
            "model_type": "use ac_pc" if self.use_ac_pc else " use combination",
            "objective_value": round(self.objective_value, 2),
            "selected_facilities": list(self.selected_facilities.keys()),
            "objective_type": objective,
            "flow_coverage": self.flow_coverage,
        }

        return result

    def _build_pulp_model(
        self,
        objective="flow",
        facility_combinations=None,
        threshold=0.0,
        weight=0.99,
        include_destination=False,
    ):
        """
        Build the PuLP model structure (variables and constraints).
        Uses AC-PC for basic/threshold models, facility combinations for capacitated.
        """
        if self.capacity is not None:
            if (
                not hasattr(self, "path_refueling_combinations")
                or not self.path_refueling_combinations
            ):
                self.generate_path_refueling_combinations(
                    method="combination", facility_combinations=facility_combinations
                )
        else:
            if not hasattr(self, "k") or not self.k:
                self.generate_path_refueling_combinations(method="ac_pc")

        model_name = f"FRLM_{objective.upper()}"
        if self.capacity is not None:
            model_name = f"Capacitated_{model_name}"

        self.model = pulp.LpProblem(model_name, pulp.LpMaximize)

        # Add facility variables
        FlowModelBuilder.add_facility_variables(
            self, candidate_sites=self.candidate_sites
        )

        # Add facility count constraint
        FacilityModelBuilder.add_facility_constraint(
            self, p_facilities=self.p_facilities
        )

        if self.use_ac_pc:
            # AC-PC approach for basic/threshold models
            FlowModelBuilder.add_flow_variables(self, flows=self.flows)
            FlowModelBuilder.add_ac_pc_constraints(
                self, a=self.a, k=self.k, candidate_sites=self.candidate_sites
            )

            flow_list = list(self.flows.items())
            if objective == "vmt":
                self._calculate_path_distances()
                self.model += pulp.lpSum(
                    flow_list[q - 1][1]
                    * self._path_distances[flow_list[q - 1][0]]
                    * self.flow_vars[q]
                    for q in self.flow_vars
                    if q <= len(flow_list)
                )
            else:
                self.model += pulp.lpSum(
                    flow_list[q - 1][1] * self.flow_vars[q]
                    for q in self.flow_vars
                    if q <= len(flow_list)
                )
        else:
            # Combination approach for capacitated model
            if self.capacity is not None:
                FlowModelBuilder.add_flow_variables(
                    self,
                    flows=self.flows,
                    path_refueling_combinations=self.path_refueling_combinations,
                )
                FlowModelBuilder.add_flow_coverage_constraints(
                    self,
                    flows=self.flows,
                    path_refueling_combinations=self.path_refueling_combinations,
                )
                FlowModelBuilder.add_capacity_constraints(
                    self,
                    flows=self.flows,
                    path_refueling_combinations=self.path_refueling_combinations,
                    e_coefficients=self.e_coefficients,
                    capacity=self.capacity,
                    candidate_sites=self.candidate_sites,
                )

                if objective == "vmt":
                    self._calculate_path_distances()

                self.model += pulp.lpSum(
                    self.flows[od_pair]
                    * (self._path_distances[od_pair] if objective == "vmt" else 1)
                    * self.flow_vars.get((q, h), 0)
                    for q, od_pair in enumerate(self.flows.keys())
                    for h in range(
                        len(self.path_refueling_combinations.get(od_pair, []))
                    )
                    if (q, h) in self.flow_vars
                )

        # Add threshold constraints if applicable (works for both AC-PC and combination)
        if threshold > 0:
            if not hasattr(self, "node_weights") or self.node_weights is None:
                self.calculate_node_weights(include_destination=include_destination)

            self.original_objective = self.model.objective

            node_coverage_vars = {}
            for origin in {od[0] for od in self.flows}:
                node_coverage_vars[origin] = pulp.LpVariable(
                    f"node_coverage_{origin}", lowBound=0, upBound=1, cat=pulp.LpBinary
                )

            # Add threshold constraints
            flow_list = list(self.flows.items())
            for origin in node_coverage_vars:
                if self.use_ac_pc:
                    # AC-PC version
                    origin_flows = [
                        q
                        for q in range(1, len(flow_list) + 1)
                        if flow_list[q - 1][0][0] == origin and q in self.flow_vars
                    ]

                    if origin_flows:
                        total_origin_flow = sum(
                            flow_list[q - 1][1] for q in origin_flows
                        )

                        self.model += (
                            pulp.lpSum(
                                flow_list[q - 1][1] * self.flow_vars[q]
                                for q in origin_flows
                            )
                            >= threshold
                            * total_origin_flow
                            * node_coverage_vars[origin]
                        )
                else:
                    # Combination version for capacitated
                    origin_flows = [
                        (q, od_pair)
                        for q, od_pair in enumerate(self.flows.keys())
                        if od_pair[0] == origin
                    ]

                    total_origin_flow = sum(
                        self.flows[od_pair] for _, od_pair in origin_flows
                    )

                    self.model += (
                        pulp.lpSum(
                            self.flows[od_pair] * self.flow_vars[(q, h)]
                            for q, od_pair in origin_flows
                            for h in range(
                                len(self.path_refueling_combinations[od_pair])
                            )
                            if (q, h) in self.flow_vars
                        )
                        >= threshold * total_origin_flow * node_coverage_vars[origin]
                    )

            # Update objective with weighted combination
            node_coverage_term = pulp.lpSum(
                self.node_weights[origin] * node_coverage_vars[origin]
                for origin in node_coverage_vars
            )

            # Set new objective
            self.model.setObjective(
                weight * node_coverage_term + (1 - weight) * self.original_objective
            )

            self.node_coverage_vars = node_coverage_vars

    def _update_pulp_variables_from_solution(self):
        """
        Update PuLP variables with solution values from greedy solver.
        """
        # Update facility variables
        for k, site in enumerate(self.candidate_sites):
            value = self.selected_facilities.get(site, 0)
            self.facility_vars[k].setInitialValue(value)

        # Update flow variables
        if self.use_ac_pc:
            # AC-PC: simple flow variables indexed by q
            flow_list = list(self.flows.items())
            for q in range(1, len(flow_list) + 1):
                if q in self.flow_vars:
                    od_pair = flow_list[q - 1][0]
                    coverage_prop = self.flow_coverage.get(od_pair, {}).get(
                        "covered_proportion", 0.0
                    )
                    self.flow_vars[q].setInitialValue(coverage_prop)

        else:
            for (q, h), var in self.flow_vars.items():
                od_list = list(self.flows.keys())
                if q < len(od_list):
                    od_pair = od_list[q]
                    coverage_prop = self.flow_coverage.get(od_pair, {}).get(
                        "covered_proportion", 0.0
                    )
                    if od_pair in self.path_refueling_combinations:
                        num_combos = len(self.path_refueling_combinations[od_pair])
                        if num_combos > 0 and h == 0:
                            var.setInitialValue(coverage_prop)
                        else:
                            var.setInitialValue(0.0)

        if hasattr(self, "node_coverage_vars") and self.node_coverage_vars is not None:
            for origin, var in self.node_coverage_vars.items():
                value = 1.0 if origin in self.covered_nodes else 0.0
                var.setInitialValue(value)

    def _evaluate_solution_with_combinations(self, facilities: dict[Any, int]) -> dict:
        """
        Evaluate how good a facility configuration is using combinations approach.
        Uses AC-PC for basic/threshold, combinations for capacitated.
        """
        if hasattr(self, "use_ac_pc") and self.use_ac_pc:
            # AC_PC evaluation
            flow_coverage = {}
            total_flow = sum(self.flows.values())
            covered_flow = 0

            flow_list = list(self.flows.items())
            for q in range(1, len(flow_list) + 1):
                if q not in self.a:
                    continue

                od_pair, flow_volume = flow_list[q - 1]

                path_covered = True
                for arc in self.a[q]:
                    arc_covered = False
                    key = (q, arc)
                    if key in self.k:
                        for node in self.k[key]:
                            if node in facilities and facilities[node] > 0:
                                arc_covered = True
                                break
                    if not arc_covered:
                        path_covered = False
                        break

                coverage = 1.0 if path_covered else 0.0
                flow_coverage[od_pair] = {
                    "flow_volume": flow_volume,
                    "covered_proportion": coverage,
                    "covered_volume": flow_volume * coverage,
                }
                covered_flow += flow_volume * coverage

            return {
                "flow_coverage": flow_coverage,
                "total_flow": total_flow,
                "covered_volume": covered_flow,
                "coverage_percentage": (
                    covered_flow / total_flow if total_flow > 0 else 0
                ),
            }

        else:
            flow_coverage = {}
            total_flow = sum(self.flows.values())
            covered_flow = 0

            if self.capacity is not None:
                # Capacitated model evaluation
                for od_pair, flow_volume in self.flows.items():
                    if od_pair not in self.path_refueling_combinations:
                        flow_coverage[od_pair] = {
                            "flow_volume": flow_volume,
                            "covered_proportion": 0.0,
                            "covered_volume": 0.0,
                        }
                        continue

                    best_coverage = 0.0

                    # Check each valid combination
                    for combination in self.path_refueling_combinations[od_pair]:
                        # Check if all facilities in combination are available
                        can_cover = True
                        min_coverage_ratio = 1.0

                        for facility in combination:
                            if facility not in facilities or facilities[facility] == 0:
                                can_cover = False
                                break

                            # For capacitated, check capacity constraints
                            # Calculate demand on this facility from this flow
                            usage_coef = (
                                2 if facility not in [od_pair[0], od_pair[1]] else 1
                            )
                            demand = (
                                self.e_coefficients.get(od_pair, 1.0)
                                * usage_coef
                                * flow_volume
                            )
                            available_capacity = facilities[facility] * self.capacity

                            if demand > 0 and available_capacity > 0:
                                coverage_ratio = min(1.0, available_capacity / demand)
                                min_coverage_ratio = min(
                                    min_coverage_ratio, coverage_ratio
                                )
                            elif demand > 0:
                                can_cover = False
                                break

                        if can_cover:
                            best_coverage = max(best_coverage, min_coverage_ratio)

                    flow_coverage[od_pair] = {
                        "flow_volume": flow_volume,
                        "covered_proportion": best_coverage,
                        "covered_volume": flow_volume * best_coverage,
                    }
                    covered_flow += flow_volume * best_coverage

            else:
                # Uncapacitated model evaluation
                open_facilities = set(facilities.keys())

                for od_pair, flow_volume in self.flows.items():
                    is_covered = False

                    if od_pair in self.path_refueling_combinations:
                        for combination in self.path_refueling_combinations[od_pair]:
                            if set(combination).issubset(open_facilities):
                                is_covered = True
                                break

                    coverage = 1.0 if is_covered else 0.0
                    flow_coverage[od_pair] = {
                        "flow_volume": flow_volume,
                        "covered_proportion": coverage,
                        "covered_volume": flow_volume * coverage,
                    }
                    covered_flow += flow_volume * coverage

            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

            return {
                "flow_coverage": flow_coverage,
                "total_flow": total_flow,
                "covered_volume": covered_flow,
                "coverage_percentage": coverage_percentage,
            }

    def _solve_pulp(
        self,
        solver_instance: pulp.LpSolver = None,
        objective: str = "flow"
    ) -> dict:
        """
        solver_instance : pulp.LpSolver, default=None
            Solver instance to use.
        """
        start_time = time.time()

        if solver_instance is not None:
            self.model.solve(solver_instance)
        else:
            raise ValueError(
                "No solver instance provided. Please specify a valid PuLP solver."
            )

        self.solution_time = round(time.time() - start_time, 2)
        self.pulp_status = self.model.status
        self.status = pulp.LpStatus[self.model.status]

        if self.model.status == pulp.LpStatusOptimal:
            # Extract selected facilities
            for k, site in enumerate(self.candidate_sites):
                if self.facility_vars[k].value() > 0.5:
                    if self.capacity is not None:
                        self.selected_facilities[site] = int(
                            round(self.facility_vars[k].value())
                        )
                    else:
                        self.selected_facilities[site] = 1

            self.objective_value = pulp.value(self.model.objective)
            result = {
                "status": self.status,
                "model_type": "ac_pc" if self.use_ac_pc else "combination",
                "objective_value": round(self.objective_value, 2),
                "selected_facilities": list(self.selected_facilities.keys()),
                "solution_time": round(self.solution_time, 2),
                "objective_type": objective,
            }
            return result

        else:
            print(f"No optimal solution found. Status: {self.status}")
            return {"status": self.status}

    def _evaluate_capacitated_solution(self, facilities: dict[Any, int]) -> dict:
        """
        Evaluate a potential facility configuration for the capacitated model.

        Parameters
        ----------
        facilities : Dict[Any, int]
            Dictionary of facilities and their potential capacities

        Returns
        -------
        Dict
            Evaluation results of the facility configuration
        """
        flow_coverage = {}
        total_flow = sum(self.flows.values())
        covered_flow = 0

        for _q, od_pair in enumerate(self.flows):
            flow_volume = self.flows[od_pair]

            valid_combinations = [
                combination
                for combination in self.path_refueling_combinations[od_pair]
                if all(facility in facilities for facility in combination)
            ]

            best_covered_proportion = 0

            for combination in valid_combinations:
                facility_usage_details = [
                    {
                        "facility": facility,
                        "usage": self.e_coefficients[od_pair]
                        * compute_facility_usage(
                            od_pair[0], od_pair[1], facility, combination
                        )
                        * flow_volume,
                        "available_capacity": self.capacity
                        * facilities.get(facility, 0),
                    }
                    for facility in combination
                ]

                try:
                    coverage_proportion = min(
                        usage_detail["available_capacity"] / usage_detail["usage"]
                        for usage_detail in facility_usage_details
                        if usage_detail["usage"] > 0
                    )

                    coverage_proportion = min(1.0, max(0.0, coverage_proportion))

                    best_covered_proportion = max(
                        best_covered_proportion, coverage_proportion
                    )

                except (ValueError, ZeroDivisionError):
                    continue

            flow_coverage[od_pair] = {
                "flow_volume": flow_volume,
                "covered_proportion": best_covered_proportion,
                "covered_volume": flow_volume * best_covered_proportion,
            }

            covered_flow += flow_volume * best_covered_proportion

        coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

        return {
            "status": "Optimal" if covered_flow > 0 else "Infeasible",
            "objective_value": covered_flow,
            "flow_coverage": flow_coverage,
            "total_flow": total_flow,
            "covered_flow": covered_flow,
            "coverage_percentage": coverage_percentage,
        }

    def _initialize_greedy_solution(self, method: str) -> set:
        supported_methods = ["empty", "random", "central", "high_flow"]

        if method not in supported_methods:
            raise ValueError(
                f"Unsupported initialisation method '{method}'. "
                f"Supported methods are: {', '.join(supported_methods)}"
            )

        if method == "empty":
            return set()

        elif method == "random":
            init_count = random.randint(1, min(2, len(self.candidate_sites)))
            return set(random.sample(self.candidate_sites, init_count))

        elif method == "central":
            if hasattr(self.network, "degree"):
                centrality = dict(self.network.degree())
                sorted_nodes = sorted(
                    self.candidate_sites,
                    key=lambda x: centrality.get(x, 0),
                    reverse=True,
                )
                init_count = min(2, len(sorted_nodes))
                return set(sorted_nodes[:init_count])
            else:
                return {self.candidate_sites[0]} if self.candidate_sites else set()

        elif method == "high_flow":
            node_importance = dict.fromkeys(self.candidate_sites, 0)

            for od_pair, volume in self.flows.items():
                if od_pair in self.flow_paths:
                    path = self.flow_paths[od_pair]
                    for node in path:
                        if node in node_importance:
                            node_importance[node] += volume

            sorted_nodes = sorted(
                self.candidate_sites, key=lambda x: node_importance[x], reverse=True
            )
            init_count = min(2, len(sorted_nodes))

            return set(sorted_nodes[:init_count])

    def _calculate_path_distances(self):
        """Calculate path distances for VMT objective."""
        self._path_distances = {}
        scipy_network = self.scipy_network.tocsr()
        for od_pair, path in self.flow_paths.items():
            distance = 0
            for i in range(len(path) - 1):
                try:
                    distance += scipy_network[path[i], path[i + 1]]
                except (KeyError, IndexError):
                    distance += 1.0
            self._path_distances[od_pair] = distance

    def get_solver_details(self) -> dict:
        """
        Retrieve detailed solver information.

        Parameters
        ----------
        verbose : bool, default True
            If True, print the details to console. If False, return details dictionary.

        Returns
        -------
        Dict
            A dictionary containing detailed solver information
        """
        if not self.solver_type:
            raise ValueError("No solver results available. Call solve() first.")

        solver_details = {
            "solver_type": self.solver_type,
            "solver_status": self.status,
        }

        if self.solver_type == "greedy":
            solver_details.update(
                {
                    "total_iterations": len(self.greedy_iterations),
                    "iteration_details": self.greedy_iterations,
                    "marginal_contributions": dict(
                        sorted(
                            self.marginal_contributions.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:10]
                    ),
                }
            )
        else:
            solver_details.update(
                {
                    "pulp_status": self.pulp_status,
                    "model_name": self.model.name if self.model else "N/A",
                    "solver_stats": self.solver_stats,
                    "lagrange_multipliers": dict(
                        list(self.lagrange_multipliers.items())[:5]
                    ),
                    "reduced_costs": dict(list(self.reduced_costs.items())[:5]),
                }
            )

        return solver_details

    @property
    def selected_facilities(self):
        """Read facility selection from PuLP variables."""
        if not self.facility_vars:
            return {}

        facilities = {}
        for k, var in enumerate(self.facility_vars):
            if var.varValue and var.varValue > 0.5:
                site = self.candidate_sites[k]
                if self.capacity is not None:
                    facilities[site] = int(round(var.varValue))
                else:
                    facilities[site] = 1
        return facilities

    def to_dataframes(self, include_iterations: bool = True) -> dict[str, pd.DataFrame]:
        """
        Export solution to pandas DataFrames.

        Parameters
        ----------
        include_iterations : bool, default=True
            Whether to include iteration details (for greedy solver)

        Returns
        -------
        dict
            Dictionary of DataFrames with keys:
            - 'facilities': Selected facilities and modules
            - 'coverage': Flow coverage details
            - 'summary': Summary statistics
            - 'iterations': Greedy solver iterations (if applicable)
            - 'shadow_prices': Shadow prices/Lagrange multipliers (if applicable)
        """
        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        if not self.flow_coverage:
            self.get_flow_coverage()

        dataframes = {}

        dataframes["facilities"] = pd.DataFrame(
            [
                {"facility_id": k, "modules": v}
                for k, v in self.selected_facilities.items()
            ]
        )

        coverage_data = []
        for od_pair, coverage in self.flow_coverage.items():
            coverage_data.append(
                {
                    "origin": od_pair[0],
                    "destination": od_pair[1],
                    "flow_volume": coverage["flow_volume"],
                    "covered_proportion": coverage["covered_proportion"],
                    "covered_volume": coverage["covered_volume"],
                }
            )
        dataframes["coverage"] = pd.DataFrame(coverage_data)

        covered_flow_sum = sum(
            coverage.get('coverage', 0)
            for coverage in self.flow_coverage.values()
        )
        total_flow = sum(self.flows.values())
        coverage_pct = (covered_flow_sum / total_flow * 100)

        summary_data = {
            "Metric": [
                "Model Type",
                "Solver Type",
                "Vehicle Range",
                "Facilities to Locate",
                "Facility Capacity",
                "Solution Status",
                "Objective Value",
                "Solution Time (s)",
                "Total Flow",
                "Covered Flow",
                "Coverage %",
            ],
            "Value": [
                "Capacitated" if self.capacity is not None else "Basic",
                self.solver_type,
                self.vehicle_range,
                self.p_facilities,
                self.capacity if self.capacity is not None else "N/A",
                self.status,
                self.objective_value,
                self.solution_time,
                sum(self.flows.values()),
                sum(
                    coverage.get("covered_volume", 0)
                    for coverage in self.flow_coverage.values()
                ),
                f"Flow Coverage: {coverage_pct:.2f}%"
            ],
        }

        dataframes["summary"] = pd.DataFrame(summary_data)

        if (
            include_iterations
            and self.solver_type == "greedy"
            and self.greedy_iterations
        ):
            iterations_data = []
            for iteration in self.greedy_iterations:
                iterations_data.append(
                    {
                        "iteration": iteration["iteration"],
                        "selected_facility": iteration["selected_facility"],
                        "objective_value": iteration["objective_value"],
                        "marginal_benefit": iteration["marginal_benefit"],
                        "candidates_evaluated": len(iteration["candidates_evaluated"]),
                    }
                )
            dataframes["iterations"] = pd.DataFrame(iterations_data)

        return dataframes

    def get_shadow_prices(self) -> dict[str, float]:
        if self.solver_type == "pulp":
            return self.shadow_prices.copy()
        elif self.solver_type == "greedy":
            shadow_prices = {}

            shadow_prices["facility_count"] = self._estimate_facility_shadow_price()

            for od_pair, coverage in self.flow_coverage.items():
                if coverage.get("covered_proportion", 0.0) > 0.99:
                    constraint_name = f"coverage_{od_pair[0]}_{od_pair[1]}"
                    shadow_prices[constraint_name] = self.flows[od_pair]

            return shadow_prices
        else:
            return {}

    def get_reduced_costs(self) -> dict[str, float]:
        if self.solver_type == "pulp":
            return self.reduced_costs.copy()
        elif self.solver_type == "greedy":
            reduced_costs = {}
            for var_name, value in self.variables.items():
                if var_name.startswith(("x_", "z_")) and value < 0.5:
                    if hasattr(self, "marginal_contributions"):
                        node_id = int(var_name.split("_")[1]) if "_" in var_name else 0
                        reduced_costs[var_name] = -self.marginal_contributions.get(
                            node_id, 0.0
                        )
                    else:
                        reduced_costs[var_name] = 0.0
            return reduced_costs
        else:
            return {}

    def get_variable_values(self) -> dict[str, float]:
        return self.variables.copy()

    def get_shadow_price(self, constraint_name: str) -> float:
        shadow_prices = self.get_shadow_prices()
        return shadow_prices.get(constraint_name, 0.0)

    def get_variable_value(self, var_name: str) -> float:
        return self.variables.get(var_name, 0.0)

    def get_slack(self, constraint_name: str) -> float:
        if constraint_name == "facility_count":
            return 0.0
        return 0.0

    def is_constraint_active(self, constraint_name: str) -> bool:
        return abs(self.get_slack(constraint_name)) < 1e-6

    def _estimate_facility_shadow_price(self) -> float:
        if not hasattr(self, "flow_coverage"):
            self.get_flow_coverage()
        best_improvement = 0.0
        current_facilities = set(self.selected_facilities.keys())

        for candidate in self.candidate_sites:
            if candidate not in current_facilities:
                improvement = 0.0
                for od_pair, coverage in self.flow_coverage.items():
                    if coverage.get("covered_proportion", 0.0) < 1.0:
                        path = self.flow_paths[od_pair]
                        if candidate in path:
                            uncovered = self.flows[od_pair] * (
                                1.0 - coverage.get("covered_proportion", 0.0)
                            )
                            improvement += uncovered

                best_improvement = max(best_improvement, improvement)

        return best_improvement

    @property
    def problem(self):
        """Return the PuLP problem instance."""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("No problem instance available. Call solve() first.")
        return self.model

    def _get_flow_details(self) -> dict:
        total_flow = sum(self.flows.values())
        covered_flow = sum(
            coverage.get("covered_volume", 0)
            for coverage in self.flow_coverage.values()
        )

        return {
            "total_flows": len(self.flows),
            "total_volume": total_flow,
            "covered_volume": covered_flow,
            "coverage_percentage": covered_flow / total_flow if total_flow > 0 else 0,
            "flow_breakdown": self.flow_coverage,
            "uncovered_flows": [
                od
                for od, cov in self.flow_coverage.items()
                if cov.get("covered_proportion", 0) < 0.01
            ],
        }

    def _get_constraint_details(self) -> dict:
        active_constraints = {
            name: info
            for name, info in self.constraints.items()
            if info.get("active", False)
        }

        return {
            "total_constraints": len(self.constraints),
            "active_constraints": len(active_constraints),
            "constraint_info": self.constraints,
            "binding_constraints": list(active_constraints.keys()),
        }

    def _get_variable_details(self) -> dict:
        facility_vars = {
            name: value
            for name, value in self.variables.items()
            if name.startswith(("x_", "z_")) and value > 1e-6
        }
        flow_vars = {
            name: value
            for name, value in self.variables.items()
            if name.startswith("y_") and value > 1e-6
        }

        return {
            "total_variables": len(self.variables),
            "facility_variables": facility_vars,
            "flow_variables": flow_vars,
            "nonzero_variables": len(
                [v for v in self.variables.values() if abs(v) > 1e-6]
            ),
        }

    def summary(self) -> dict:
        """
        Generate a summary of the solution.
        """
        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        if not hasattr(self, "flow_coverage"):
            self.get_flow_coverage()

        if self.threshold > 0 and not hasattr(self, "covered_nodes"):
            self.calculate_covered_nodes()

        if not hasattr(self, "solver_stats"):
            self.extract_solver_statistics()

        total_flow = sum(self.flows.values())
        covered_flow = sum(
            coverage.get("covered_volume", 0)
            for coverage in self.flow_coverage.values()
        )
        coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

        fully_covered = sum(
            1
            for coverage in self.flow_coverage.values()
            if coverage["covered_proportion"] >= 0.99
        )
        partially_covered = sum(
            1
            for coverage in self.flow_coverage.values()
            if 0 < coverage["covered_proportion"] < 0.99
        )
        uncovered = sum(
            1
            for coverage in self.flow_coverage.values()
            if coverage["covered_proportion"] <= 0.01
        )

        total_modules = (
            sum(self.selected_facilities.values())
            if self.capacity is not None
            else len(self.selected_facilities)
        )

        summary_dict = {
            "model": {
                "type": "Capacitated" if self.capacity is not None else "Basic",
                "solver": self.solver_type,
                "vehicle_range": self.vehicle_range,
                "facilities_to_locate": self.p_facilities,
                "facility_capacity": self.capacity,
                "coverage_threshold": self.threshold,
                "weight_parameter": self.weight,
            },
            "solution": {
                "status": self.status,
                "objective_value": self.objective_value,
                "solution_time": self.solution_time,
            },
            "facilities": {
                "total_modules": total_modules,
                "locations": list(self.selected_facilities.keys()),
                "details": [
                    {"site": facility, "modules": count}
                    for facility, count in self.selected_facilities.items()
                ],
            },
            "flow_coverage": {
                "total_flow": total_flow,
                "covered_flow": covered_flow,
                "coverage_percentage": coverage_percentage,
                "flows_breakdown": {
                    "fully_covered": f"{fully_covered} / {len(self.flows)}",
                    "partially_covered": f"{partially_covered} / {len(self.flows)}",
                    "uncovered": f"{uncovered} / {len(self.flows)}",
                },
            },
            "solver_information": self.solver_stats,
        }

        if self.threshold > 0 and hasattr(self, "covered_nodes"):
            total_origins = len({od[0] for od in self.flows})
            summary_dict["node_coverage"] = {
                "covered_nodes": len(self.covered_nodes),
                "total_origins": total_origins,
                "coverage_percentage": len(self.covered_nodes) / total_origins,
            }

        return summary_dict

    def __repr__(self):
        if (
            isinstance(self._original_vehicle_range, float)
            and 0 < self._original_vehicle_range < 1
        ):
            range_str = (
                f"vehicle_range={self._original_vehicle_range:.0%} of longest path"
            )
        else:
            range_str = f"vehicle_range={self.vehicle_range}"

        capacity_info = (
            f", capacity={self.capacity}" if self.capacity is not None else ""
        )
        threshold_info = f", threshold={self.threshold}" if self.threshold > 0 else ""
        weight_info = f", weight={self.weight}" if self.threshold > 0 else ""

        return (
            f"FRLM({range_str}, p={self.p_facilities}{capacity_info}"
            f"{threshold_info}{weight_info})"
        )

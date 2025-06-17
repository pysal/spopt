import numpy as np
import networkx as nx
import pulp
import time
import itertools
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Iterable, Literal
from tqdm import tqdm
import warnings
import random
import json
import logging
import scipy.sparse as sp
import libpysal
import scipy.sparse.csgraph as csgraph

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


class FRLM:
    """
        Flow Refueling Location Model (FRLM)

        Parameters
        ----------
        vehicle_range : Union[float, int], default=200000.0
            Defines the maximum travel distance or range for vehicles.
            - If value is between 0 and 1: Treated as a percentage of the longest path. e.g. 0.5: 50% of the longest path in the network
            - If value > 1: Treated as an absolute distance in distance units (e.g., meters)

        p_facilities : int, default=5
            Number of facility modules to be located in the network.
        
        capacity : Union[float, None], default=None
            Facility module capacity constraint.
            - If None: Uses an uncapacitated model
            - If provided: Sets a maximum capacity for each facility module

        threshold : float, default=0.0
            Minimum flow coverage percentage for a node to be considered covered, 0 disables threshold extension.

        weight : float, default=0.99
            Controls the trade-off between node coverage and flow coverage for threshold extension objective.
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
        vehicle_range: Union[float, int] = 200000,
        p_facilities: int = 5,
        capacity: Union[float, None] = None,
        threshold: float = 0.0,
        weight: float = 0.99,
        objective: str = "flow",
        include_destination: bool = False
    ):
        self.vehicle_range = vehicle_range
        self._original_vehicle_range = vehicle_range
        self.p_facilities = p_facilities

        self.capacity = capacity

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

        self.K = {}  # K[path_id, arc_id] = [refueling_nodes]
        self.a = {}  # a[path_id] = [arc_ids]

        self.node_weights = None

        self.selected_facilities = {}
        self.flow_coverage = {}
        self.covered_nodes = []
        self.objective_value = 0
        self.solution_time = 0
        self.status = None

        self.solver_type = None
        self.variables = {}  
        self.constraints = {}

        self.model = None
        self.lagrange_multipliers = {}  
        self.reduced_costs = {} 
        self.shadow_prices = {}

        self.greedy_iterations = []
        self.marginal_contributions = {}

        self.solver_stats = {}

        self.bounds = {"lower": None, "upper": None}

        
    def get_variable_value(self, var_name: str) -> Optional[float]:
        return self.variables.get(var_name)

    def get_constraint_dual(self, constraint_name: str) -> Optional[float]:
        return self.lagrange_multipliers.get(constraint_name)

    def get_reduced_cost(self, var_name: str) -> Optional[float]:
        return self.reduced_costs.get(var_name)
    
    @classmethod
    def convert_to_scipy_sparse(cls, network):
        """
        Convert various network formats (networkx, libpysal.graph, numpy arrays) to scipy sparse matrix.

        Parameters
        ----------
        network : object
            Input network 

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if isinstance(network, nx.Graph):
            return nx.to_scipy_sparse_array(
                network, 
                weight='weight',  
                format='csr'
            )

        if isinstance(network, libpysal.graph.Graph):
            return network.sparse

        if sp.issparse(network):
            return network

        if isinstance(network, np.ndarray):
            return sp.csr_matrix(network)
        
        else:
            raise TypeError(f"Unsupported network type: {type(network)}, expected networkx.Graph, libpysal.graph.Graph, numpy.ndarray, or scipy.sparse matrix.")

    @classmethod
    def from_flow_dataframe(cls, 
                  network, 
                  flows: Union[pd.DataFrame, Dict], 
                  **kwargs):
        """
        Create FRLM instance from network and flows.

        Parameters
        ----------
        network : object
            Network representation
        flows : Union[pd.DataFrame, Dict]
            Flow data
        """
        scipy_network = cls.convert_to_scipy_sparse(network)
        frlm = cls(**kwargs)
        
        frlm.scipy_network = scipy_network
        frlm.network = network
        frlm.candidate_sites = list(range(scipy_network.shape[0]))

        if flows is not None:
            if isinstance(flows, dict):
                frlm.load_flows_from_dict(flows)
            elif isinstance(flows, pd.DataFrame):
                required_columns = ['origin', 'destination', 'volume']
                if not all(col in flows.columns for col in required_columns):
                    raise ValueError(f"Flows DataFrame must contain columns: {required_columns}")
                
                frlm.load_flows_from_dataframe(flows)
            else:
                raise TypeError("Flows must be a dictionary or DataFrame")
        else:
            warnings.warn("No flows provided. FRLM will not have any flow data to work with.", UserWarning)

        return frlm

    def set_candidate_sites(self, sites: List) -> None:
        for site in sites:
            if site not in self.network.nodes():
                raise KeyError(f"Site {site} is not a node in the network")
        self.candidate_sites = sites
    
    def add_network(self, network):
        """
        Add network to the FRLM instance.
        
        Parameters
        ----------
        network : object
            Network representation to be converted to scipy sparse matrix
        
        Returns
        -------
        self : FRLM
            Returns the instance for method chaining
        """
        self.scipy_network = self.convert_to_scipy_sparse(network)
        self.network = network
        self.candidate_sites = list(range(self.scipy_network.shape[0]))
        
        return self

    def add_flows(self, flows):
        """
        Add multiple flows to the FRLM instance.
        
        Parameters
        ----------
        flows : Union[pd.DataFrame, Dict[Tuple[Any, Any], float]]
            Flows to be added
        
        Returns
        -------
        self : FRLM
            Returns the instance for method chaining
        """
        if self.scipy_network is None:
            raise ValueError("Network must be added before adding flows")

        if isinstance(flows, dict):
            self.load_flows_from_dict(flows)
        elif isinstance(flows, pd.DataFrame):
            self.load_flows_from_dataframe(flows)
        else:
            raise TypeError("Flows must be a dictionary or DataFrame")
        
        return self

    def add_flow(
        self, origin: Any, destination: Any, volume: float, path: Optional[List] = None
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
        if origin < 0 or origin >= num_nodes or destination < 0 or destination >= num_nodes:
            raise KeyError(f"Origin {origin} or destination {destination} is not in the network")

        if path is not None:
            self.flows[(origin, destination)] = volume
            self.flow_paths[(origin, destination)] = path
            self._set_adaptive_vehicle_range()
            return

        try:
            distances, predecessors = csgraph.shortest_path(
                self.scipy_network, 
                directed=False, 
                indices=origin, 
                return_predecessors=True
            )

            if np.isinf(distances[destination]):
                raise KeyError(f"No path exists between {origin} and {destination}")

            def reconstruct_path(distances, predecessors, origin, destination):
                if predecessors[destination] == -9999:
                    raise KeyError(f"No path exists between {origin} and {destination}")

                path = [destination]
                current = destination

                while current != origin:
                    prev = predecessors[current]
                    if prev == -9999:
                        raise KeyError(f"No path exists between {origin} and {destination}")

                    path.insert(0, prev)

                    current = prev

                return path

            path = reconstruct_path(distances, predecessors, origin, destination)
            if path[0] != origin or path[-1] != destination:
                raise KeyError(f"Path reconstruction failed for {origin} to {destination}")

        except Exception as e:
            print(f"{e}")
            raise ValueError(f"Error finding path between {origin} and {destination}: {e}")

        self.flows[(origin, destination)] = volume
        self.flow_paths[(origin, destination)] = path

        self._set_adaptive_vehicle_range()

    def load_flows_from_dict(self, flows: Dict[Tuple[Any, Any], float], compute_paths: bool = True) -> None:
        """
        Load flows from dictionary by converting to DataFrame and using load_flows_from_dataframe().
        
        Parameters
        ----------
        flows : Dict[Tuple[Any, Any], float]
            Dictionary mapping (origin, destination) tuples to flow volumes
        compute_paths : bool, default=True
            Whether to compute shortest paths for each flow
        """

        data = []
        for (origin, destination), volume in flows.items():
            data.append({
                'origin': origin,
                'destination': destination, 
                'volume': volume
            })
        
        df = pd.DataFrame(data)

        self.load_flows_from_dataframe(
            df, 
            origin_col='origin',
            destination_col='destination', 
            volume_col='volume',
            path_col=None
        )

    def load_flows_from_dataframe(
        self,
        df: pd.DataFrame,
        origin_col: str = "origin",
        destination_col: str = "destination",
        volume_col: str = "volume",
        path_col: Optional[str] = None,
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
            if path_col and path_col in row:
                path = row[path_col]

                if not isinstance(path, list):
                    raise TypeError(
                        f"Path must be a list of integer node ids. "
                    )

                if path:
                    if path[0] != origin or path[-1] != destination:
                        raise ValueError(
                            f"Path for OD pair ({origin}, {destination}) is inconsistent. "
                            f"Path should start at {origin} and end at {destination}."
                        )

            self.add_flow(origin, destination, volume, path)

        self._set_adaptive_vehicle_range()

    def _set_adaptive_vehicle_range(self):

        """Set vehicle range based on percentage of longest path.
        """
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

    def _generate_k_a_sets(self) -> Tuple[Dict, Dict]:
        """
        Generate K and A sets for basic FRLM model using scipy network.

        - K[path_id, arc_id]: 
            Set of candidate refueling nodes for a specific path segment
        - A[path_id]: 
            Set of arc/segment indices for a given path
        """
        
        if self.scipy_network is None:
            raise ValueError("No network loaded")

        distances = csgraph.shortest_path(self.scipy_network, method='auto')

        self.K = {}
        self.a = {}

        for path_id, (od_pair, path) in enumerate(self.flow_paths.items(), 1):
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

                self.K[(path_id, arc_id)] = refueling_nodes

            self.a[path_id] = arc_ids

        return self.K, self.a

    @staticmethod
    def rising_combination(values: List, start: int = 1, stop: int = None) -> Iterable[List]:
    
        """
        Generate combinations of increasing sizes from a list of values.
        
        Parameters
        ----------
        values : list
            Input list to generate combinations from
        start : int, optional
            Minimum size of combinations (default is 1)
        stop : int or None, optional
            Maximum size of combinations
        
        Yields
        ------
        List
            Combinations of different sizes
        """
        if stop is None:
            stop = len(values)
        
        if start < 1:
            raise ValueError("Start must be at least 1")
        if stop > len(values):
            stop = len(values)
        
        for size in range(start, min(stop + 1, len(values) + 1)):
            yield from map(list, itertools.combinations(values, size))


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

    def compute_facility_usage(
        self, 
        origin: Any, 
        destination: Any, 
        facility: Any, 
        combination: List
    ) -> int:
        """
        Compute facility usage coefficient for capacitated model.

        Parameters
        ----------
        origin : Any
            Origin node of the flow
        destination : Any
            Destination node of the flow
        facility : Any
            Facility being evaluated
        combination : List
            List of facilities in the current combination

        Returns
        -------
        int
            Facility usage coefficient:
            - 0: Facility is not in the combination or not used
            - 1: Facility is at origin/destination or used once
            - 2: Facility is used twice (intermediate refueling), account for round trip usage 
        """
        # Facility not in the combination
        if facility not in combination:
            return 0

        # Facility is at origin or destination
        if facility == origin or facility == destination:
            return 1

        # Facility is used as an intermediate refueling point
        return 2

    def check_path_refueling_feasibility(self, path: List, facilities: List) -> bool:
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

        distances = csgraph.shortest_path(self.scipy_network, method='auto')

        total_length = sum(
            distances[path[i], path[i + 1]]
            for i in range(len(path) - 1)
        )

        if total_length * 2 <= self.vehicle_range:
            return any(facility in path for facility in facilities)

        remaining_range = self.vehicle_range / 2
        if path[0] in facilities:
            remaining_range = self.vehicle_range

        for i in range(1, len(path)):
            segment_length = distances[path[i-1], path[i]]
            remaining_range -= segment_length

            if remaining_range < 0:
                return False

            if path[i] in facilities:
                remaining_range = self.vehicle_range

        remaining_range = self.vehicle_range / 2
        if path[-1] in facilities:
            remaining_range = self.vehicle_range

        for i in range(len(path) - 1, 0, -1):
            segment_length = distances[path[i-1], path[i]]
            remaining_range -= segment_length

            if remaining_range < 0:
                return False

            if path[i-1] in facilities:
                remaining_range = self.vehicle_range

        return True

    def generate_path_refueling_combinations(
        self, 
        facility_combinations: Optional[List[List]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Dict:
        """
        Generate dictionary mapping OD pairs to valid facility combinations
        
        Parameters
        ----------
        facility_combinations : Optional[List[List]], optional
            Pre-computed facility combinations
        start : int, optional
            Minimum size of combinations
            If None, defaults to 1
        stop : int, optional
            Maximum size of combinations
            If None, defaults to p_facilities
        
        Returns
        -------
        Dict
            Dictionary of OD pairs to valid facility combinations
        """

        if facility_combinations is not None:
            self.facility_combinations = facility_combinations
        else:
            start = start if start is not None else 1
            stop = stop if stop is not None else self.p_facilities

            self.facility_combinations = list(
                self.rising_combination(
                    self.candidate_sites, 
                    start=start, 
                    stop=stop
                )
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

    def calculate_node_weights(
        self, 
        include_destination: bool = False
    ) -> np.ndarray:
        """
           Calculate node weights based on flow volumes.

        Parameters
        ----------
        include_destination : bool, default=False
            If True, include destination nodes in weight calculation.
            If False, only use origin nodes for weight calculation, as specified in the paper (Hong and Kuby, 2016).

        Returns
        -------
        np.ndarray
            Array of node weights
        """
        if not self.flows:
            raise ValueError("Flows must be loaded first")

        max_node = max(max(od[0], od[1]) for od in self.flows.keys())
        node_weights = np.zeros(max_node + 1)
        total_flow = sum(self.flows.values())

        for (origin, destination), volume in self.flows.items():

            node_weights[origin] += volume

            if include_destination:
                node_weights[destination] += volume

        node_weights = (
            node_weights / (total_flow * (2 if include_destination else 1))
        )

        self.node_weights = node_weights
        return node_weights

    def solve(
        self,
        solver:  Union[Literal["greedy"], pulp.LpSolver] = None,
        seed: Optional[int] = None,
        initialization_method: str = "empty",
        max_iterations: int = 100,
        facility_combinations: Optional[List[List]] = None, 
        **kwargs,
    ) -> Dict:
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

        threshold = kwargs.get('threshold', self.threshold)
        weight = kwargs.get('weight', self.weight)

        objective = kwargs.get('objective', self.objective)
        include_destination = kwargs.get('include_destination', self.include_destination)

        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        if weight < 0 or weight > 1:
            raise ValueError(f"Weight must be between 0 and 1, got {weight}")
        
        self.threshold = threshold
        self.weight = weight

        if self.threshold > 0:
            self.calculate_node_weights(
                include_destination= include_destination
            )

        if not self.network:
            raise ValueError("No network loaded.")

        if not self.flows:
            raise ValueError("No flows loaded. ")

        self._clear_solution_data()

        if solver == "greedy":
            self.solver_type = "greedy"
            return self._solve_greedy(
                objective=objective,
                seed=seed,
                initialization_method=initialization_method,
                max_iterations=max_iterations,
                facility_combinations=facility_combinations,
                **kwargs,
            )

        else:
            self.solver_type = type(solver).__name__
            return self._solve_pulp(
                solver_instance=solver,
                objective=objective,
                facility_combinations=facility_combinations,
                **kwargs,
            )

    def _clear_solution_data(self):
        self.variables = {}
        self.constraints = {}
        self.lagrange_multipliers = {}
        self.reduced_costs = {}
        self.shadow_prices = {}
        self.greedy_iterations = []
        self.marginal_contributions = {}
        self.solver_stats = {}
        self.bounds = {"lower": None, "upper": None}
        self.model = None
        self.pulp_status = None

    def _solve_greedy(
        self,
        objective: str = "flow",
        weight_method: str = "double",
        seed: Optional[int] = None,
        initialization_method: str = "empty",
        max_iterations: int = 100,
        facility_combinations: Optional[List[List]] = None,
        **kwargs,
    ) -> Dict:
        """
        Solve using greedy heuristic for facility location.

        Greedy Solver :
        1. Initialisation Phase:
            - Start with an empty set of facilities or a small initial set
            - Goal: Incrementally build a solution by adding facilities
            - Each iteration aims to maximise the objective function

        2. Objective Function Evaluation:
            At each iteration, evaluate each candidate facility by tentatively adding it to the current solution.
            Measure how much it improves the objective:
                - For flow: sum of flow volumes covered.
                - For vmt: flow volume × path distance.
                - If a threshold is set: consider both node coverage and flow coverage using a weighted combination.

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

        path_distances = {}
        if objective == "vmt":
            for od_pair, path in self.flow_paths.items():
                distance = sum(
                    self.network[path[i]][path[i + 1]]["length"]
                    for i in range(len(path) - 1)
                )
                path_distances[od_pair] = distance

        if self.capacity is not None:

            if not self.path_refueling_combinations:
                self.generate_path_refueling_combinations(facility_combinations=facility_combinations)
            # Initialise solution tracking
            current_facilities = {} 
            current_objective = 0
            iterations = []
            marginal_contributions = {}
             # Main greedy selection loop
            for iteration in range(self.p_facilities):
                best_location = None
                best_improvement = 0
                iteration_candidates = {}
                # Evaluate each candidate facility
                for candidate_facility in self.candidate_sites:
                    # Skip if facility is already at maximum
                    candidate_facilities = current_facilities.copy()
                    candidate_facilities[candidate_facility] = (
                        candidate_facilities.get(candidate_facility, 0) + 1
                    )
                    # Evaluate solution with candidate facility
                    evaluation = self._evaluate_capacitated_solution(
                        candidate_facilities
                    )

                    if evaluation["status"] == "Optimal":
                        # Compute objective value improvement
                        if objective == "flow":
                            obj_value = evaluation.get("objective_value", 0.0)
                            if obj_value is None:
                                obj_value = 0.0
                            improvement = obj_value - current_objective
                        elif objective == "vmt":
                            # VMT-specific improvement calculation
                            if evaluation.get("flow_coverage"):
                                vmt_covered = sum(
                                    evaluation["flow_coverage"][od_pair].get(
                                        "covered_proportion", 0.0
                                    )
                                    * self.flows[od_pair]
                                    * path_distances[od_pair]
                                    for od_pair in evaluation["flow_coverage"]
                                )
                                improvement = vmt_covered - current_objective
                            else:
                                improvement = 0.0
                         # Track candidate information
                        iteration_candidates[candidate_facility] = {
                            "facility_id": candidate_facility,
                            "objective_value": evaluation["objective_value"],
                            "marginal_benefit": improvement,
                        }
                        marginal_contributions[candidate_facility] = improvement
                         # Update best location if improvement is better
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_location = candidate_facility
                # Record iteration details
                iteration_info = {
                    "iteration": iteration + 1,
                    "selected_facility": best_location,
                    "objective_value": current_objective + best_improvement,
                    "candidates_evaluated": iteration_candidates,
                    "marginal_benefit": best_improvement,
                }
                iterations.append(iteration_info)
                # Stop if no improvement found
                if best_location is None:
                    break
                  # Update current solution
                current_facilities[best_location] = (
                    current_facilities.get(best_location, 0) + 1
                )
                current_objective += best_improvement

            self.selected_facilities = current_facilities
            self.greedy_iterations = iterations
            self.marginal_contributions = marginal_contributions

            final_evaluation = self._evaluate_capacitated_solution(current_facilities)
            self.flow_coverage = final_evaluation["flow_coverage"]

            if objective == "flow":
                self.objective_value = final_evaluation.get("objective_value", 0.0)
            elif objective == "vmt":
                if final_evaluation.get("flow_coverage"):
                    try:
                        self.objective_value = sum(
                            final_evaluation["flow_coverage"][od_pair].get(
                                "covered_proportion", 0.0
                            )
                            * self.flows[od_pair]
                            * path_distances.get(od_pair, 0.0)
                            for od_pair in final_evaluation["flow_coverage"]
                        )
                    except Exception as e:
                        print(f"Error calculating VMT objective: {e}")
                        self.objective_value = 0.0
                else:
                    self.objective_value = 0.0
            else:
                self.objective_value = 0.0

            if self.objective_value is None:
                self.objective_value = 0.0

            self.solution_time = time.time() - start_time
            self.status = "Heuristic"

            self.solver_stats = {
                "total_candidates_evaluated": sum(
                    len(iter_info["candidates_evaluated"]) for iter_info in iterations
                ),
                "average_candidates_per_iteration": np.mean(
                    [len(iter_info["candidates_evaluated"]) for iter_info in iterations]
                ),
                "convergence_pattern": [
                    iter_info["objective_value"] for iter_info in iterations
                ],
                "marginal_benefits": [
                    iter_info["marginal_benefit"] for iter_info in iterations
                ],
            }

            result = final_evaluation.copy()
            result.update(
                {
                    "objective_value": self.objective_value,
                    "solution_time": self.solution_time,
                    "status": self.status,
                    "objective_type": objective,
                }
            )

            if objective == "vmt":
                total_vmt = sum(
                    self.flows[od_pair] * path_distances[od_pair]
                    for od_pair in self.flows
                )
                covered_vmt = self.objective_value
                vmt_coverage_percentage = (
                    covered_vmt / total_vmt if total_vmt > 0 else 0
                )

                result.update(
                    {
                        "total_vmt": total_vmt,
                        "covered_vmt": covered_vmt,
                        "vmt_coverage_percentage": vmt_coverage_percentage,
                    }
                )

            return result

        else:

            if not self.K or not self.a:
                self._generate_k_a_sets()

            num_facilities = (
                max([node for key in self.K for node in self.K[key]]) if self.K else 0
            )
            num_flows = len(self.a)

            z = np.zeros(num_facilities + 1, dtype=int)  # Facility locations
            y = np.zeros(num_flows + 1, dtype=int)  # Covered flows
            c = None  # Covered nodes (for threshold extension)

            iterations = []
            marginal_contributions = {}

            if self.threshold > 0:
                # self.calculate_node_weights(method=weight_method)
                c = np.zeros(len(self.node_weights), dtype=int)

                origin_flows = {}
                origin_total_flow = {}

                for q in range(1, num_flows + 1):
                    if q in self.a:
                        for od_pair in self.flow_paths:
                            if list(self.flow_paths.keys()).index(od_pair) == q - 1:
                                origin = od_pair[0]
                                if origin not in origin_flows:
                                    origin_flows[origin] = []
                                    origin_total_flow[origin] = 0
                                origin_flows[origin].append(q)
                                origin_total_flow[origin] += self.flows[od_pair]
                                break
             # Initialisation step
            initial_facilities = self._initialize_greedy_solution(initialization_method)

            for facility in initial_facilities:
                if 1 <= facility <= num_facilities:
                    z[facility] = 1

            if initial_facilities:
                for q in range(1, num_flows + 1):
                    if q not in self.a or y[q] == 1:
                        continue

                    path_covered = True
                    for arc in self.a[q]:
                        arc_covered = False
                        key = (q, arc)
                        if key in self.K:
                            for node in self.K[key]:
                                if z[node] == 1:
                                    arc_covered = True
                                    break
                        if not arc_covered:
                            path_covered = False
                            break

                    if path_covered:
                        y[q] = 1

                if self.threshold > 0 and c is not None:
                    for origin in origin_flows:
                        if origin_total_flow[origin] > 0:
                            covered_flow = sum(
                                list(self.flows.values())[q - 1]
                                for q in origin_flows[origin]
                                if y[q] == 1
                            )
                            if (
                                covered_flow
                                >= self.threshold * origin_total_flow[origin]
                            ):
                                c[origin] = 1

            remaining_facilities = self.p_facilities - len(initial_facilities)
            iteration_count = 0

            for facility_count in range(remaining_facilities):
                iteration_count += 1

                if iteration_count > max_iterations:
                    break

                best_facility = None
                best_value = 0
                iteration_candidates = {}

                for candidate_facility in range(1, num_facilities + 1):
                    if z[candidate_facility] == 1:
                        continue

                    z[candidate_facility] = 1

                    new_y = y.copy()
                    for q in range(1, num_flows + 1):
                        if q not in self.a or new_y[q] == 1:
                            continue

                        path_covered = True
                        for arc in self.a[q]:
                            arc_covered = False
                            key = (q, arc)
                            if key in self.K:
                                for node in self.K[key]:
                                    if z[node] == 1:
                                        arc_covered = True
                                        break
                            if not arc_covered:
                                path_covered = False
                                break

                        if path_covered:
                            new_y[q] = 1

                    if self.threshold > 0:
                        new_c = np.zeros(len(self.node_weights), dtype=int)
                        for origin in origin_flows:
                            if origin_total_flow[origin] > 0:
                                covered_flow = sum(
                                    list(self.flows.values())[q - 1]
                                    for q in origin_flows[origin]
                                    if new_y[q] == 1
                                )
                                if (
                                    covered_flow
                                    >= self.threshold * origin_total_flow[origin]
                                ):
                                    new_c[origin] = 1

                        weighted_coverage = sum(
                            self.node_weights[j] * new_c[j] for j in range(len(new_c))
                        )
                        flow_coverage = sum(
                            list(self.flows.values())[q - 1] * new_y[q]
                            for q in range(1, len(new_y))
                            if q <= len(self.flows)
                        ) / sum(self.flows.values())
                        if objective == "vmt":
                            vmt_coverage = sum(
                                list(self.flows.values())[q - 1]
                                * path_distances[list(self.flows.keys())[q - 1]]
                                * new_y[q]
                                for q in range(1, len(new_y))
                                if q <= len(self.flows)
                                and new_y[q] == 1
                                and list(self.flows.keys())[q - 1] in path_distances
                            ) / sum(
                                self.flows[od] * path_distances[od]
                                for od in self.flows.keys()
                            )
                            evaluation = (
                                self.weight * weighted_coverage
                                + (1 - self.weight) * vmt_coverage
                            )
                        else:
                            flow_coverage = sum(
                                list(self.flows.values())[q - 1] * new_y[q]
                                for q in range(1, len(new_y))
                                if q <= len(self.flows)
                            ) / sum(self.flows.values())
                            evaluation = (
                                self.weight * weighted_coverage
                                + (1 - self.weight) * flow_coverage
                            )
                    else:
                        if objective == "vmt":
                            evaluation = sum(
                                list(self.flows.values())[q - 1]
                                * path_distances[list(self.flows.keys())[q - 1]]
                                for q in range(1, len(new_y))
                                if q <= len(self.flows)
                                and new_y[q] == 1
                                and list(self.flows.keys())[q - 1] in path_distances
                            )
                        else:
                            evaluation = sum(
                                list(self.flows.values())[q - 1]
                                for q in range(1, len(new_y))
                                if q <= len(self.flows) and new_y[q] == 1
                            )

                    z[candidate_facility] = 0

                    marginal_benefit = evaluation - (
                        self.objective_value if self.objective_value else 0
                    )
                    iteration_candidates[candidate_facility] = {
                        "facility_id": candidate_facility,
                        "objective_value": evaluation,
                        "marginal_benefit": marginal_benefit,
                    }
                    marginal_contributions[candidate_facility] = marginal_benefit

                    if evaluation > best_value:
                        best_value = evaluation
                        best_facility = candidate_facility

                iteration_info = {
                    "iteration": len(initial_facilities) + facility_count + 1,
                    "selected_facility": best_facility,
                    "objective_value": best_value,
                    "candidates_evaluated": iteration_candidates,
                    "marginal_benefit": iteration_candidates.get(best_facility, {}).get(
                        "marginal_benefit", 0
                    ),
                }
                iterations.append(iteration_info)

                if best_facility is not None:
                    z[best_facility] = 1

                    for q in range(1, num_flows + 1):
                        if q not in self.a or y[q] == 1:
                            continue

                        path_covered = True
                        for arc in self.a[q]:
                            arc_covered = False
                            key = (q, arc)
                            if key in self.K:
                                for node in self.K[key]:
                                    if z[node] == 1:
                                        arc_covered = True
                                        break
                            if not arc_covered:
                                path_covered = False
                                break

                        if path_covered:
                            y[q] = 1

                    if self.threshold > 0 and c is not None:
                        for origin in origin_flows:
                            if origin_total_flow[origin] > 0:
                                covered_flow = sum(
                                    list(self.flows.values())[q - 1]
                                    for q in origin_flows[origin]
                                    if y[q] == 1
                                )
                                if (
                                    covered_flow
                                    >= self.threshold * origin_total_flow[origin]
                                ):
                                    c[origin] = 1

            self.solution_time = time.time() - start_time
            self.status = "Heuristic"
            self.greedy_iterations = iterations
            self.marginal_contributions = marginal_contributions

            self.constraints = {}
            self.constraints["facility_count"] = {
                "name": "facility_count",
                "type": "eq",
                "rhs": float(self.p_facilities),
                "slack": 0.0,
                "shadow_price": self._estimate_facility_shadow_price(),
                "active": True,
            }

            for od_pair, coverage in self.flow_coverage.items():
                if coverage.get("covered_proportion", 0.0) > 0.99:
                    constraint_name = f"coverage_{od_pair[0]}_{od_pair[1]}"
                    self.constraints[constraint_name] = {
                        "name": constraint_name,
                        "type": "geq",
                        "rhs": 1.0,
                        "slack": 0.0,
                        "shadow_price": self.flows[od_pair],
                        "active": True,
                    }

            for site in self.candidate_sites:
                is_selected = site in self.selected_facilities
                self.variables[f"x_{site}"] = 1.0 if is_selected else 0.0

            for i, od_pair in enumerate(self.flows.keys()):
                if i + 1 <= len(y) and y[i + 1] == 1:
                    self.variables[f"y_{od_pair[0]}_{od_pair[1]}"] = 1.0
                else:
                    self.variables[f"y_{od_pair[0]}_{od_pair[1]}"] = 0.0

            self.selected_facilities = {
                i: 1 for i in range(1, num_facilities + 1) if z[i] == 1
            }

            covered_flows = [
                q for q in range(1, num_flows + 1) if q in self.a and y[q] == 1
            ]
            total_flow_covered = sum(
                list(self.flows.values())[q - 1]
                for q in covered_flows
                if q <= len(self.flows)
            )
            coverage_percentage = (
                total_flow_covered / sum(self.flows.values())
                if sum(self.flows.values()) > 0
                else 0
            )

            if self.threshold > 0 and c is not None:
                self.covered_nodes = [j for j in range(len(c)) if c[j] == 1]
                weighted_coverage = sum(
                    self.node_weights[j] * c[j] for j in range(len(c))
                )
                self.objective_value = (
                    self.weight * weighted_coverage
                    + (1 - self.weight) * coverage_percentage
                )
            else:
                self.covered_nodes = []
                self.objective_value = total_flow_covered

            self.flow_coverage = {}
            for i, od_pair in enumerate(self.flows.keys()):
                if i + 1 <= len(y) and y[i + 1] == 1:
                    self.flow_coverage[od_pair] = {
                        "flow_volume": self.flows[od_pair],
                        "covered_proportion": 1.0,
                        "covered_volume": self.flows[od_pair],
                    }
                else:
                    self.flow_coverage[od_pair] = {
                        "flow_volume": self.flows[od_pair],
                        "covered_proportion": 0.0,
                        "covered_volume": 0.0,
                    }

            self.solver_stats = {
                "total_candidates_evaluated": sum(
                    len(iter_info["candidates_evaluated"]) for iter_info in iterations
                ),
                "average_candidates_per_iteration": (
                    np.mean(
                        [
                            len(iter_info["candidates_evaluated"])
                            for iter_info in iterations
                        ]
                    )
                    if iterations
                    else 0
                ),
                "convergence_pattern": [
                    iter_info["objective_value"] for iter_info in iterations
                ],
                "marginal_benefits": [
                    iter_info["marginal_benefit"] for iter_info in iterations
                ],
                "initialization_method": initialization_method,
                "initial_facilities": list(initial_facilities),
                "seed_used": seed,
            }

            if not hasattr(self, "bounds"):
                self.bounds = {}
            self.bounds["upper"] = self.objective_value
            self.bounds["lower"] = self.objective_value

            return {
                "status": self.status,
                "objective_value": self.objective_value,
                "selected_facilities": self.selected_facilities,
                "flow_coverage": self.flow_coverage,
                "covered_nodes": self.covered_nodes,
                "total_flow": sum(self.flows.values()),
                "covered_flow": total_flow_covered,
                "coverage_percentage": coverage_percentage,
                "solution_time": self.solution_time,
                "initialization_method": initialization_method,
                "initial_facilities": list(initial_facilities),
                "seed_used": seed,
                "iterations_completed": len(iterations),
                "solver_stats": self.solver_stats,
            }

    def _solve_pulp(
        self,
        solver_instance: pulp.LpSolver = None,
        objective: str = "flow",
        weight_method: str = "double",
        facility_combinations: Optional[List[List]] = None,
        **kwargs,
    ) -> Dict:
        """
        solver_instance : pulp.LpSolver, default=None
            Solver instance to use. 
        """
        if self.capacity is not None:
            if not self.path_refueling_combinations:
                self.generate_path_refueling_combinations(facility_combinations=facility_combinations)

            start_time = time.time()
            path_distances = {}
            if objective == "vmt":
                for od_pair, path in self.flow_paths.items():
                    distance = sum(
                        self.network[path[i]][path[i + 1]]["length"]
                        for i in range(len(path) - 1)
                    )
                    path_distances[od_pair] = distance

            model_name = (
                "Capacitated_FRLM_VMT" if objective == "vmt" else "Capacitated_FRLM"
            )
            model = pulp.LpProblem(model_name, pulp.LpMaximize)
            self.model = model

            x = {
                k: pulp.LpVariable(f"x_{k}", lowBound=0, cat=pulp.LpInteger)
                for k in self.candidate_sites
            }

            y = {}
            for q, od_pair in enumerate(self.flows.keys()):
                valid_combinations = self.path_refueling_combinations[od_pair]
                for h, combination in enumerate(valid_combinations):
                    y[(q, h)] = pulp.LpVariable(
                        f"y_{q}_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )

            constraint_refs = {}

            if objective == "vmt":
                model += pulp.lpSum(
                    self.flows[od_pair] * path_distances[od_pair] * y[(q, h)]
                    for q, od_pair in enumerate(self.flows.keys())
                    for h, combination in enumerate(
                        self.path_refueling_combinations[od_pair]
                    )
                )
            else:
                model += pulp.lpSum(
                    self.flows[od_pair] * y[(q, h)]
                    for q, od_pair in enumerate(self.flows.keys())
                    for h, combination in enumerate(
                        self.path_refueling_combinations[od_pair]
                    )
                )

            for k in self.candidate_sites:
                capacity_constraint = (
                    pulp.lpSum(
                        self.e_coefficients[od_pair]
                        * self.compute_facility_usage(
                            od_pair[0], od_pair[1], k, combination
                        )
                        * self.flows[od_pair]
                        * y[(q, h)]
                        for q, od_pair in enumerate(self.flows.keys())
                        for h, combination in enumerate(
                            self.path_refueling_combinations[od_pair]
                        )
                        if k in combination
                    )
                    <= self.capacity * x[k]
                )
                model += capacity_constraint
                constraint_refs[f"capacity_{k}"] = capacity_constraint

            module_constraint = pulp.lpSum(x.values()) == self.p_facilities
            model += module_constraint
            constraint_refs["module_count"] = module_constraint

            for q, od_pair in enumerate(self.flows.keys()):
                flow_constraint = (
                    pulp.lpSum(
                        y[(q, h)]
                        for h in range(len(self.path_refueling_combinations[od_pair]))
                    )
                    <= 1
                )
                model += flow_constraint
                constraint_refs[f"flow_coverage_{q}"] = flow_constraint

        else:
            if not self.K or not self.a:
                self._generate_k_a_sets()

            start_time = time.time()
            model = pulp.LpProblem("Basic_FRLM", pulp.LpMaximize)
            self.model = model

            num_facilities = (
                max([node for key in self.K for node in self.K[key]]) if self.K else 0
            )
            num_flows = len(self.a)

            z = {
                i: pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary)
                for i in range(1, num_facilities + 1)
            }
            y = {
                q: pulp.LpVariable(f"y_{q}", cat=pulp.LpBinary)
                for q in range(1, num_flows + 1)
                if q in self.a
            }

            constraint_refs = {}

            path_distances = {}
            if objective == "vmt":
                for od_pair, path in self.flow_paths.items():
                    distance = sum(
                        self.network[path[i]][path[i + 1]]["length"]
                        for i in range(len(path) - 1)
                    )
                    path_distances[od_pair] = distance

            if self.threshold > 0:
                # self.calculate_node_weights(method=weight_method)
                c = {
                    j: pulp.LpVariable(f"c_{j}", cat=pulp.LpBinary)
                    for j in range(1, len(self.node_weights))
                    if self.node_weights[j] > 0
                }

                for j in c:
                    flows_for_node = []
                    flow_sum = 0
                    for i, od_pair in enumerate(self.flows.keys()):
                        if od_pair[0] == j:
                            q = i + 1
                            if q in y:
                                flows_for_node.append(q)
                                flow_sum += self.flows[od_pair]

                    if flow_sum > 0 and flows_for_node:
                        covered_flow = pulp.lpSum(
                            self.flows[list(self.flows.keys())[q - 1]] * y[q]
                            for q in flows_for_node
                            if q in y
                        )
                        constraint = covered_flow >= self.threshold * flow_sum * c[j]
                        model += constraint
                        constraint_refs[f"threshold_node_{j}"] = constraint

                if objective == "vmt":
                    # VMT-based objective with threshold
                    vmt_term = pulp.lpSum(
                        self.flows[list(self.flows.keys())[q - 1]]
                        * path_distances[list(self.flows.keys())[q - 1]]
                        * y[q]
                        for q in y
                        if q <= len(self.flows)
                        and list(self.flows.keys())[q - 1] in path_distances
                    )
                    model += self.weight * pulp.lpSum(
                        self.node_weights[j] * c[j] for j in c
                    ) + (1 - self.weight) * vmt_term / sum(
                        self.flows[od] * path_distances[od] for od in self.flows.keys()
                    )
                else:
                    # Flow-based objective with threshold
                    model += self.weight * pulp.lpSum(
                        self.node_weights[j] * c[j] for j in c
                    ) + (1 - self.weight) * pulp.lpSum(
                        list(self.flows.values())[q - 1] * y[q]
                        for q in y
                        if q <= len(self.flows)
                    ) / sum(
                        self.flows.values()
                    )
            else:
                if objective == "vmt":
                    model += pulp.lpSum(
                        self.flows[list(self.flows.keys())[q - 1]]
                        * path_distances[list(self.flows.keys())[q - 1]]
                        * y[q]
                        for q in y
                        if q <= len(self.flows)
                        and list(self.flows.keys())[q - 1] in path_distances
                    )
                else:
                    model += pulp.lpSum(
                        list(self.flows.values())[q - 1] * y[q]
                        for q in y
                        if q <= len(self.flows)
                    )

            for q in y:
                if q in self.a:
                    for arc in self.a[q]:
                        key = (q, arc)
                        if key in self.K:
                            refuel_facilities = pulp.lpSum(
                                z[node] for node in self.K[key] if node in z
                            )
                            constraint = refuel_facilities >= y[q]
                            model += constraint
                            constraint_refs[f"flow_coverage_{q}_{arc}"] = constraint

            facility_constraint = pulp.lpSum(z.values()) == self.p_facilities
            model += facility_constraint
            constraint_refs["facility_count"] = facility_constraint

        if solver_instance is not None:
            model.solve(solver_instance)
        else:
            raise ValueError(
                "No solver instance provided. Please specify a valid PuLP solver."
            )

        self.solution_time = time.time() - start_time
        self.pulp_status = model.status
        self.status = pulp.LpStatus[model.status]

        if (
            model.status == pulp.LpStatusOptimal
            or model.status == pulp.LpStatusNotSolved
        ):
            self.objective_value = pulp.value(model.objective)

            for var in model.variables():
                self.variables[var.name] = var.varValue

            if self.capacity is not None:
                self.constraints = {}
                for constraint_name, constraint in constraint_refs.items():
                    slack_val = getattr(constraint, "slack", None)
                    shadow_val = getattr(constraint, "pi", None)

                    slack_val = slack_val if slack_val is not None else 0.0
                    shadow_val = shadow_val if shadow_val is not None else 0.0

                    self.constraints[constraint_name] = {
                        "name": constraint_name,
                        "type": (
                            "eq"
                            if "=" in str(constraint)
                            else "leq" if "<=" in str(constraint) else "geq"
                        ),
                        "rhs": float(self.p_facilities),
                        "slack": slack_val,
                        "shadow_price": shadow_val,
                        "active": abs(slack_val) < 1e-6,
                    }

            try:
                for constraint_name, constraint in constraint_refs.items():
                    if hasattr(constraint, "pi") and constraint.pi is not None:
                        self.lagrange_multipliers[constraint_name] = constraint.pi
                        self.shadow_prices[constraint_name] = constraint.pi
            except AttributeError:
                print("Warning: Dual values not available with current solver")

            try:
                for var in model.variables():
                    if hasattr(var, "dj") and var.dj is not None:
                        self.reduced_costs[var.name] = var.dj
            except AttributeError:
                print("Warning: Reduced costs not available with current solver")

            if self.capacity is not None:
                self.selected_facilities = {
                    k: int(x[k].value())
                    for k in self.candidate_sites
                    if x[k].value() > 0
                }

                self.flow_coverage = {}
                total_flow = sum(self.flows.values())
                covered_flow = 0

                for q, od_pair in enumerate(self.flows.keys()):
                    flow_volume = self.flows[od_pair]
                    covered_proportion = sum(
                        y[(q, h)].value()
                        for h in range(len(self.path_refueling_combinations[od_pair]))
                        if y[(q, h)].value() > 0
                    )

                    self.flow_coverage[od_pair] = {
                        "flow_volume": flow_volume,
                        "covered_proportion": covered_proportion,
                        "covered_volume": flow_volume * covered_proportion,
                    }

                    if objective == "vmt":
                        self.flow_coverage[od_pair]["distance"] = path_distances[
                            od_pair
                        ]
                        self.flow_coverage[od_pair]["covered_vmt"] = (
                            flow_volume * path_distances[od_pair] * covered_proportion
                        )

                    covered_flow += flow_volume * covered_proportion

            else:
                self.selected_facilities = {i: 1 for i in z if pulp.value(z[i]) > 0.5}

                covered_flows = [q for q in y if pulp.value(y[q]) > 0.5]
                total_flow = sum(self.flows.values())
                covered_flow = sum(
                    list(self.flows.values())[q - 1]
                    for q in covered_flows
                    if q <= len(self.flows)
                )

                self.flow_coverage = {}
                for i, od_pair in enumerate(self.flows.keys()):
                    if i + 1 in y and pulp.value(y[i + 1]) > 0.5:
                        self.flow_coverage[od_pair] = {
                            "flow_volume": self.flows[od_pair],
                            "covered_proportion": 1.0,
                            "covered_volume": self.flows[od_pair],
                        }
                    else:
                        self.flow_coverage[od_pair] = {
                            "flow_volume": self.flows[od_pair],
                            "covered_proportion": 0.0,
                            "covered_volume": 0.0,
                        }

                if self.threshold > 0:
                    self.covered_nodes = [j for j in c if pulp.value(c[j]) > 0.5]
                else:
                    self.covered_nodes = []

            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

            self.solver_stats = {
                "solver_name": str(solver_instance),
                "num_variables": len(model.variables()),
                "num_constraints": len(
                    [constraint for constraint in model.constraints.values()]
                ),
                "objective_sense": "maximize",
            }

            result = {
                "status": self.status,
                "objective_value": self.objective_value,
                "selected_facilities": self.selected_facilities,
                "flow_coverage": self.flow_coverage,
                "total_flow": total_flow,
                "covered_flow": covered_flow,
                "coverage_percentage": coverage_percentage,
                "solution_time": self.solution_time,
            }

            if self.capacity is None and self.threshold > 0:
                result["covered_nodes"] = self.covered_nodes

            if self.capacity is not None and objective == "vmt":
                total_vmt = sum(
                    self.flows[od_pair] * path_distances[od_pair]
                    for od_pair in self.flows
                )
                covered_vmt = sum(
                    coverage.get("covered_vmt", 0)
                    for coverage in self.flow_coverage.values()
                )
                vmt_coverage_percentage = (
                    covered_vmt / total_vmt if total_vmt > 0 else 0
                )

                result.update(
                    {
                        "total_vmt": total_vmt,
                        "covered_vmt": covered_vmt,
                        "vmt_coverage_percentage": vmt_coverage_percentage,
                        "objective_type": objective,
                    }
                )

            return result
        else:
            print(f"No optimal solution found. Status: {self.status}")
            return {"status": self.status}

    def _evaluate_capacitated_solution(self, facilities: Dict[Any, int]) -> Dict:
        model = pulp.LpProblem("Evaluate_CFRLM", pulp.LpMaximize)

        y = {}
        for q, od_pair in enumerate(self.flows.keys()):
            valid_combinations = self.path_refueling_combinations[od_pair]
            for h, combination in enumerate(valid_combinations):
                if all(facility in facilities for facility in combination):
                    y[(q, h)] = pulp.LpVariable(
                        f"y_{q}_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous
                    )

        model += pulp.lpSum(
            self.flows[od_pair] * y[(q, h)]
            for q, od_pair in enumerate(self.flows.keys())
            for h, combination in enumerate(self.path_refueling_combinations[od_pair])
            if (q, h) in y
        )

        for k in facilities.keys():
            model += (
                pulp.lpSum(
                    self.e_coefficients[od_pair]
                    * self.compute_facility_usage(
                        od_pair[0], od_pair[1], k, combination
                    )
                    * self.flows[od_pair]
                    * y[(q, h)]
                    for q, od_pair in enumerate(self.flows.keys())
                    for h, combination in enumerate(
                        self.path_refueling_combinations[od_pair]
                    )
                    if (q, h) in y and k in combination
                )
                <= self.capacity * facilities[k]
            )

        for q, od_pair in enumerate(self.flows.keys()):
            model += (
                pulp.lpSum(
                    y[(q, h)]
                    for h in range(len(self.path_refueling_combinations[od_pair]))
                    if (q, h) in y
                )
                <= 1
            )

        model.solve(pulp.GUROBI_CMD(msg=False))

        if model.status == pulp.LpStatusOptimal:
            objective_value = pulp.value(model.objective)

            flow_coverage = {}
            total_flow = sum(self.flows.values())
            covered_flow = 0

            for q, od_pair in enumerate(self.flows.keys()):
                flow_volume = self.flows[od_pair]

                covered_proportion = sum(
                    y[(q, h)].value()
                    for h in range(len(self.path_refueling_combinations[od_pair]))
                    if (q, h) in y and y[(q, h)].value() > 0
                )

                flow_coverage[od_pair] = {
                    "flow_volume": flow_volume,
                    "covered_proportion": covered_proportion,
                    "covered_volume": flow_volume * covered_proportion,
                }

                covered_flow += flow_volume * covered_proportion

            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

            return {
                "status": "Optimal",
                "objective_value": objective_value,
                "flow_coverage": flow_coverage,
                "total_flow": total_flow,
                "covered_flow": covered_flow,
                "coverage_percentage": coverage_percentage,
            }
        else:
            return {
                "status": pulp.LpStatus[model.status],
                "objective_value": 0.0,
                "flow_coverage": {},
                "total_flow": sum(self.flows.values()),
                "covered_flow": 0.0,
                "coverage_percentage": 0.0,
            }

    def _initialize_greedy_solution(self, method: str) -> set:

        supported_methods = ['empty', 'random', 'central', 'high_flow']
    
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
                return set([self.candidate_sites[0]]) if self.candidate_sites else set()

        elif method == "high_flow":
            node_importance = {node: 0 for node in self.candidate_sites}

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

    def get_solver_details(self, verbose: bool = True) -> Dict:
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
            solver_details.update({
                "total_iterations": len(self.greedy_iterations),
                "iteration_details": self.greedy_iterations,
                "marginal_contributions": dict(
                    sorted(
                        self.marginal_contributions.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]  
                )
            })
        else:  
            solver_details.update({
                "pulp_status": self.pulp_status,
                "model_name": self.model.name if self.model else 'N/A',
                "solver_stats": self.solver_stats,
                "lagrange_multipliers": dict(list(self.lagrange_multipliers.items())[:5]),
                "reduced_costs": dict(list(self.reduced_costs.items())[:5])
            })

        return solver_details

    def write_csv(self, filename: str, include_iterations: bool = True) -> None:
        """
        Export solution to CSV files.

        Parameters
        ----------
        filename : str
            Base filename for CSV export. Multiple files will be created with suffixes.
        include_iterations : bool, default True
            Whether to include solver iteration details in export.

        Raises
        ------
        ValueError
            If no solution is available.
        """
        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        facilities_df = pd.DataFrame(
            [{"facility_id": k, "modules": v} for k, v in self.selected_facilities.items()]
        )
        facilities_df.to_csv(filename.replace(".csv", "_facilities.csv"), index=False)

        coverage_data = []
        for od_pair, coverage in self.flow_coverage.items():
            coverage_data.append({
                "origin": od_pair[0],
                "destination": od_pair[1],
                "flow_volume": coverage["flow_volume"],
                "covered_proportion": coverage["covered_proportion"],
                "covered_volume": coverage["covered_volume"],
            })
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df.to_csv(filename.replace(".csv", "_coverage.csv"), index=False)

        if include_iterations:
            if self.solver_type == "greedy" and self.greedy_iterations:
                iterations_data = []
                for iteration in self.greedy_iterations:
                    iterations_data.append({
                        "iteration": iteration["iteration"],
                        "selected_facility": iteration["selected_facility"],
                        "objective_value": iteration["objective_value"],
                        "marginal_benefit": iteration["marginal_benefit"],
                        "candidates_evaluated": len(iteration["candidates_evaluated"]),
                    })
                iterations_df = pd.DataFrame(iterations_data)
                iterations_df.to_csv(
                    filename.replace(".csv", "_greedy_iterations.csv"), index=False
                )
            else:
                if self.lagrange_multipliers:
                    duals_data = []
                    for constraint_name, dual_value in self.lagrange_multipliers.items():
                        duals_data.append({
                            "constraint_name": constraint_name,
                            "lagrange_multiplier": dual_value,
                        })
                    duals_df = pd.DataFrame(duals_data)
                    duals_df.to_csv(
                        filename.replace(".csv", "_lagrange_multipliers.csv"), index=False
                    )


    def write_json(self, filename: str) -> None:
        """
        Export solution to a JSON file.

        Parameters
        ----------
        filename : str
            Filename for JSON export.

        Raises
        ------
        ValueError
            If no solution is available.
        """
        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        solution_data = {
            "model_type": "capacitated" if self.capacity is not None else "basic",
            "solver_type": self.solver_type,
            "parameters": {
                "vehicle_range": self.vehicle_range,
                "p_facilities": self.p_facilities,
                "capacity": self.capacity if self.capacity is not None else None,
                "threshold": self.threshold,
                "weight": self.weight,
            },
            "solution": {
                "status": self.status,
                "objective_value": self.objective_value,
                "solution_time": self.solution_time,
                "selected_facilities": self.selected_facilities,
                "flow_coverage": {str(k): v for k, v in self.flow_coverage.items()},
                "covered_nodes": (
                    self.covered_nodes if hasattr(self, "covered_nodes") else []
                ),
            },
            "solver_details": {
                "greedy_iterations": (
                    self.greedy_iterations if self.solver_type == "greedy" else None
                ),
                "marginal_contributions": (
                    self.marginal_contributions if self.solver_type == "greedy" else None
                ),
                "lagrange_multipliers": (
                    self.lagrange_multipliers if self.solver_type != "greedy" else None
                ),
                "reduced_costs": (
                    self.reduced_costs if self.solver_type != "greedy" else None
                ),
                "solver_stats": self.solver_stats,
            },
        }

        with open(filename, "w") as f:
            json.dump(solution_data, f, indent=2)


    def write_excel(self, filename: str, include_iterations: bool = True) -> None:
        """
        Export solution to an Excel file with multiple sheets.

        Parameters
        ----------
        filename : str
            Filename for Excel export.
        include_iterations : bool, default True
            Whether to include solver iteration details in export.

        Raises
        ------
        ValueError
            If no solution is available.
        """
        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        with pd.ExcelWriter(filename) as writer:
            facilities_df = pd.DataFrame(
                [{"facility_id": k, "modules": v} for k, v in self.selected_facilities.items()]
            )
            facilities_df.to_excel(writer, sheet_name="Facilities", index=False)

            coverage_data = []
            for od_pair, coverage in self.flow_coverage.items():
                coverage_data.append({
                    "origin": od_pair[0],
                    "destination": od_pair[1],
                    "flow_volume": coverage["flow_volume"],
                    "covered_proportion": coverage["covered_proportion"],
                    "covered_volume": coverage["covered_volume"],
                })
            coverage_df = pd.DataFrame(coverage_data)
            coverage_df.to_excel(writer, sheet_name="Flow_Coverage", index=False)

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
                    f"{sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values()) / sum(self.flows.values()) * 100:.2f}%",
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            if include_iterations:
                if self.solver_type == "greedy" and self.greedy_iterations:
                    iterations_data = []
                    for iteration in self.greedy_iterations:
                        iterations_data.append({
                            "iteration": iteration["iteration"],
                            "selected_facility": iteration["selected_facility"],
                            "objective_value": iteration["objective_value"],
                            "marginal_benefit": iteration["marginal_benefit"],
                            "candidates_evaluated": len(iteration["candidates_evaluated"]),
                        })
                    iterations_df = pd.DataFrame(iterations_data)
                    iterations_df.to_excel(
                        writer, sheet_name="Greedy_Iterations", index=False
                    )

                elif self.solver_type == "pulp":
                    if self.lagrange_multipliers:
                        duals_data = []
                        for constraint_name, dual_value in self.lagrange_multipliers.items():
                            duals_data.append({
                                "constraint_name": constraint_name,
                                "lagrange_multiplier": dual_value,
                            })
                        duals_df = pd.DataFrame(duals_data)
                        duals_df.to_excel(
                            writer, sheet_name="Lagrange_Multipliers", index=False
                        )

                    if self.reduced_costs:
                        costs_data = []
                        for var_name, reduced_cost in self.reduced_costs.items():
                            costs_data.append({
                                "variable_name": var_name,
                                "reduced_cost": reduced_cost,
                            })
                        costs_df = pd.DataFrame(costs_data)
                        costs_df.to_excel(
                            writer, sheet_name="Reduced_Costs", index=False
                        )


    def get_shadow_prices(self) -> Dict[str, float]:
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

    def get_reduced_costs(self) -> Dict[str, float]:
        if self.solver_type == "pulp":
            return self.reduced_costs.copy()
        elif self.solver_type == "greedy":
            reduced_costs = {}
            for var_name, value in self.variables.items():
                if (
                    var_name.startswith(("x_", "z_")) and value < 0.5
                ):  
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

    def get_variable_values(self) -> Dict[str, float]:
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

    def get_reduced_cost(self, var_name: str) -> float:
        if self.solver_type == "pulp":
            return self.reduced_costs.get(var_name, 0.0)
        else:
            return 0.0

    def _estimate_facility_shadow_price(self) -> float:
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

    def get_shadow_prices(self, verbose: bool = True) -> Dict:

        if self.solver_type == "pulp":
            shadow_prices = self.shadow_prices.copy()
        elif self.solver_type == "greedy":
            shadow_prices = {}
            shadow_prices["facility_count"] = self._estimate_facility_shadow_price()

            for od_pair, coverage in self.flow_coverage.items():
                if coverage.get("covered_proportion", 0.0) > 0.99:
                    constraint_name = f"coverage_{od_pair[0]}_{od_pair[1]}"
                    shadow_prices[constraint_name] = self.flows[od_pair]
        else:
            shadow_prices = {}

        filtered_shadow_prices = {
            name: price for name, price in shadow_prices.items() 
            if abs(price) > 1e-6
        }

        return filtered_shadow_prices
    @property
    def problem(self):
        if self.solver_type == "pulp":
            return self.model
        elif self.solver_type == "greedy":
            return self._create_greedy_problem_view()
        else:
            raise ValueError(
                f"Unknown solver type '{self.solver_type}'. Must be 'pulp' or 'greedy'."
            )

    def _create_greedy_problem_view(self):
        class GreedyProblemView:
            def __init__(self, frlm_instance):
                self.frlm = frlm_instance
                self.name = f"Greedy_{frlm_instance.__class__.__name__}"
                self.status = 1 if frlm_instance.status == "Heuristic" else 0

            @property
            def variables(self):
                return [
                    GreedyVariable(name, value)
                    for name, value in self.frlm.variables.items()
                ]

            @property
            def constraints(self):
                return {
                    name: GreedyConstraint(name, info)
                    for name, info in self.frlm.constraints.items()
                }

            def objective_value(self):
                return self.frlm.objective_value

        return GreedyProblemView(self)

    def get_solution_details(self) -> Dict:
        return {
            "solver_type": self.solver_type,
            "model_type": "capacitated" if self.capacity is not None else "basic",
            "problem_instance": self.problem,
            "optimisation_status": self.status,
            "objective_value": self.objective_value,
            "solution_time": self.solution_time,
            "facilities": self._get_facility_details(),
            "flows": self._get_flow_details(),
            "constraints": self._get_constraint_details(),
            "variables": self._get_variable_details(),
            "lagrange_multipliers": self.get_shadow_prices(),
            "reduced_costs": self.get_reduced_costs(),
            "solver_stats": self.solver_stats,
        }

    def _get_facility_details(self) -> Dict:
        return {
            "selected_facilities": self.selected_facilities,
            "total_modules": (
                sum(self.selected_facilities.values())
                if self.capacity is not None
                else len(self.selected_facilities)
            ),
            "locations": list(self.selected_facilities.keys()),
            "capacity_utilization": (
                self._get_capacity_utilization() if self.capacity is not None else None
            ),
        }

    def _get_flow_details(self) -> Dict:
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

    def _get_constraint_details(self) -> Dict:
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

    def _get_variable_details(self) -> Dict:
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

    def _get_capacity_utilization(self) -> Dict:
        if self.capacity is None:
            return None

        utilization = {}
        for site, modules in self.selected_facilities.items():
            total_capacity = modules * self.capacity
            used_capacity = self._calculate_facility_utilization(site)
            utilization[site] = {
                "modules": modules,
                "total_capacity": total_capacity,
                "used_capacity": used_capacity,
                "utilization_rate": (
                    used_capacity / total_capacity if total_capacity > 0 else 0
                ),
            }

        return utilization

    def summary(self) -> Dict:
        """
        Generate a summary of the solution.

        Returns
        -------
        Dict
            A dictionary containing solution details

        Raises
        ------
        ValueError
            If no solution is available.
        """

        if not self.selected_facilities:
            raise ValueError("No solution available. Call solve() first.")

        total_flow = sum(self.flows.values())
        covered_flow = sum(
            coverage.get("covered_volume", 0)
            for coverage in self.flow_coverage.values()
        )
        coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0

        fully_covered = sum(
            1 for coverage in self.flow_coverage.values()
            if coverage["covered_proportion"] >= 0.99
        )
        partially_covered = sum(
            1 for coverage in self.flow_coverage.values()
            if 0 < coverage["covered_proportion"] < 0.99
        )
        uncovered = sum(
            1 for coverage in self.flow_coverage.values()
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
                "weight_parameter": self.weight
            },
            "solution": {
                "status": self.status,
                "objective_value": self.objective_value,
                "solution_time": self.solution_time
            },
            "facilities": {
                "total_modules": total_modules,
                "locations": list(self.selected_facilities.keys()),
                "details": [
                    {"site": facility, "modules": count}
                    for facility, count in self.selected_facilities.items()
                ]
            },
            "flow_coverage": {
                "total_flow": total_flow,
                "covered_flow": covered_flow,
                "coverage_percentage": coverage_percentage,
                "flows_breakdown": {
                    "fully_covered": f"{fully_covered} / {len(self.flows)}",
                    "partially_covered": f"{partially_covered} / {len(self.flows)}",
                    "uncovered": f"{uncovered} / {len(self.flows)}"
                }
            },
            "solver_information": self.solver_stats
        }

        if hasattr(self, "covered_nodes") and self.covered_nodes:
            total_origins = len(set(od[0] for od in self.flows.keys()))
            summary_dict["node_coverage"] = {
                "covered_nodes": len(self.covered_nodes),
                "total_origins": total_origins,
                "coverage_percentage": len(self.covered_nodes) / total_origins
            }
        return summary_dict
    
    def __repr__(self):
        if isinstance(self._original_vehicle_range, float) and 0 < self._original_vehicle_range < 1:
            range_str = f"vehicle_range={self._original_vehicle_range:.0%} of longest path"
        else:
            range_str = f"vehicle_range={self.vehicle_range}"

        capacity_info = f", capacity={self.capacity}" if self.capacity is not None else ""
        threshold_info = f", threshold={self.threshold}" if self.threshold > 0 else ""
        
        return (
            f"FRLM({range_str}, p={self.p_facilities}{capacity_info}"
            f"{threshold_info}, weight={self.weight})"
        )
    
    

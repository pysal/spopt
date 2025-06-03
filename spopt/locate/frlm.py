import numpy as np
import networkx as nx
import pulp
import time
import itertools
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from tqdm import tqdm

class GreedyVariable:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.varValue = value  #
        self.dj = 0.0  

class GreedyConstraint:
    def __init__(self, name, info):
        self.name = name
        self.pi = info.get('shadow_price', 0.0)  
        self.slack = info.get('slack', 0.0)


class FRLM:
    """ 
    vehicle_range : float, default= 200000.0, with option to be n% of longest path
    p : int, default=5
        Number of facility modules to locate
    capacity : float, default=100.0
        Capacity of each facility module (for capacitated model)
    capacitated : bool, default=False
        Whether to use capacitated model constraints
    threshold : float, default=0.0
        Threshold for node coverage (0 disables threshold extension)
    weight : float, default=0.99
        Weight parameter for threshold extension objective (w)
    """
    
    def __init__(self, 
             vehicle_range: Union[float, str] = 200000.0, 
             p: int = 5, 
             capacity: float = 100.0,
             capacitated: bool = False,
             threshold: float = 0.0,
             weight: float = 0.99):

        self.vehicle_range = vehicle_range
        self._original_vehicle_range = vehicle_range 
        self.p = p
        self.capacity = capacity
        self.capacitated = capacitated
        self.threshold = threshold
        self.weight = weight

        self.network = None
        self.flows = {}
        self.flow_paths = {}
        self.candidate_sites = []

        self.facility_combinations = []
        self.path_refueling_combinations = {}
        self.e_coefficients = {}  

        self.K = {}  # K[path_id, arc_id] = [refueling_nodes]
        self.a = {}  # a[path_id] = [arc_ids]
        
        #  for threshold extension
        self.node_weights = None

        self.selected_facilities = {}
        self.flow_coverage = {}
        self.covered_nodes = []
        self.objective_value = 0
        self.solution_time = 0
        self.status = None

        self.solver_type = None
        self.variables = {}  # {var_name: value}
        self.constraints = {}
        
        # PuLP
        self.model = None 
        self.lagrange_multipliers = {}  # {constraint_name: dual_value}
        self.reduced_costs = {}  # {variable_name: reduced_cost}
        self.shadow_prices = {}  

        self.greedy_iterations = []  
        self.marginal_contributions = {}  

        self.solver_stats = {}

        self.bounds = {'lower': None, 'upper': None}
        
        if capacitated:
            print(f"Initialised FRLM with vehicle_range={vehicle_range}, p={p}, "
                f"capacity={capacity}, capacitated={capacitated}")
        else:
            print(f"Initialised FRLM with vehicle_range={vehicle_range}, p={p}, "
          f"capacitated={capacitated} (unlimited capacity)")
        if isinstance(vehicle_range, str) and vehicle_range.endswith('%'):
            print(f"Vehicle range set to {vehicle_range} of longest path after loading flows")
    
    def get_variable_value(self, var_name: str) -> Optional[float]:
        return self.variables.get(var_name)
    
    def get_constraint_dual(self, constraint_name: str) -> Optional[float]:
        return self.lagrange_multipliers.get(constraint_name)
    
    def get_reduced_cost(self, var_name: str) -> Optional[float]:
        return self.reduced_costs.get(var_name)
    
    def load_network(self, 
                    network: nx.Graph, 
                    weight_attr: str = 'length') -> None:
 
        self.network = network.copy()
        for u, v, data in self.network.edges(data=True):
            if weight_attr not in data:
                print(f"Warning: Edge ({u}, {v}) has no {weight_attr} attribute. Default setting to 100.0")
                self.network[u][v][weight_attr] = 100.0
        self.candidate_sites = list(self.network.nodes())
        
        print(f"Loaded network with {len(self.network.nodes())} nodes and {len(self.network.edges())} edges")
    
    def set_candidate_sites(self, sites: List) -> None:

        for site in sites:
            if site not in self.network.nodes():
                raise ValueError(f"Site {site} is not a node in the network")
        self.candidate_sites = sites
        print(f"Set {len(sites)} candidate sites for facility location")
    
    def add_flow(self, 
                origin: Any, 
                destination: Any, 
                volume: float, 
                path: Optional[List] = None) -> None:
        """
        origin : Origin node ID
        destination :  Destination node ID
        volume : float, Flow volume
        path : Optional[List], default=None, Specific path to use (if None, shortest path is computed)
        """
        if origin not in self.network.nodes():
            raise ValueError(f"Origin {origin} is not a node in the network")
        if destination not in self.network.nodes():
            raise ValueError(f"Destination {destination} is not a node in the network")
        
        od_pair = (origin, destination)
        self.flows[od_pair] = volume
        if path is None:
            try:
                path = nx.shortest_path(self.network, origin, destination, weight='length')
            except nx.NetworkXNoPath:
                raise ValueError(f"No path exists between {origin} and {destination}")
        
        self.flow_paths[od_pair] = path
        self._set_adaptive_vehicle_range()
    
    def load_flows_from_dict(self, 
                           flows: Dict[Tuple[Any, Any], float], 
                           compute_paths: bool = True) -> None:

        for (origin, destination), volume in flows.items():
            if compute_paths:
                self.add_flow(origin, destination, volume)
            else:
                if origin not in self.network.nodes():
                    raise ValueError(f"Origin {origin} is not a network node")
                if destination not in self.network.nodes():
                    raise ValueError(f"Destination {destination} is not a network node")
                
                self.flows[(origin, destination)] = volume
        
        print(f"Loaded {len(flows)} flows")
        self._set_adaptive_vehicle_range()
    
    def load_flows_from_dataframe(self, 
                                df: pd.DataFrame,
                                origin_col: str = 'origin',
                                destination_col: str = 'destination', 
                                volume_col: str = 'volume',
                                path_col: Optional[str] = None) -> None:
        """
        origin_col : str, default='origin'
            Column name for origin nodes
        destination_col : str, default='destination'
            Column name for destination nodes
        volume_col : str, default='volume'
            Column name for flow volumes
        path_col : Optional[str], default=None
            Column name for pre-computed paths (if available)
        """
        for _, row in df.iterrows():
            origin = row[origin_col]
            destination = row[destination_col]
            volume = row[volume_col]
            
            path = None
            if path_col and path_col in row:
                path = row[path_col]
                if isinstance(path, str):
                    path = eval(path) if path.startswith('[') else [int(x) for x in path.split(',')]
            
            self.add_flow(origin, destination, volume, path)

        self._set_adaptive_vehicle_range()
    
    def _set_adaptive_vehicle_range(self):
        """Set vehicle range based on percentage of longest path."""
        if isinstance(self._original_vehicle_range, str) and self._original_vehicle_range.endswith('%'):

            percentage = float(self._original_vehicle_range.rstrip('%')) / 100.0

            max_distance = 0
            for path in self.flow_paths.values():
                distance = sum(self.network[path[i]][path[i+1]]['length'] 
                            for i in range(len(path) - 1))
                max_distance = max(max_distance, distance)
            
            self.vehicle_range = max_distance * percentage
        else:
            self.vehicle_range = float(self._original_vehicle_range)

    def generate_k_a_sets(self) -> Tuple[Dict, Dict]:
        """
        Generate K and A sets for basic FRLM model.
        K[path_id, arc_id] contains refueling nodes for each arc on each path.
        A[path_id] contains arc IDs for each path.
        
        Returns:
        Tuple[Dict, Dict]
            K and A sets
        """
        if not self.network or not self.flows:
            raise ValueError("Network and flows must be loaded first")
        
        self.K = {}
        self.a = {}
        
        for path_id, (od_pair, path) in enumerate(self.flow_paths.items(), 1):
            arc_ids = []
            
            for arc_id, i in enumerate(range(len(path) - 1), 1):
                u, v = path[i], path[i + 1]
                arc_ids.append(arc_id)

                refueling_nodes = []
                for candidate in self.candidate_sites:
                    try:
                        dist_to_u = nx.shortest_path_length(self.network, candidate, u, weight='length')
                        dist_to_v = nx.shortest_path_length(self.network, candidate, v, weight='length')
                        
                        if (dist_to_u <= self.vehicle_range / 2 or 
                            dist_to_v <= self.vehicle_range / 2):
                            refueling_nodes.append(candidate)
                    except nx.NetworkXNoPath:
                        continue
                
                self.K[(path_id, arc_id)] = refueling_nodes
            
            self.a[path_id] = arc_ids
        
        print(f"Generated K set with {len(self.K)} entries and A set with {len(self.a)} paths")
        return self.K, self.a
    
    def generate_facility_combinations(self) -> List[List]:
        """
        Generate all possible facility combinations for capacitated model.
        """
        all_combinations = []

        all_combinations.extend([[site] for site in self.candidate_sites])

        if self.p > 1:
            for r in range(2, min(self.p + 1, len(self.candidate_sites) + 1)):
                combs = list(itertools.combinations(self.candidate_sites, r))
                all_combinations.extend([list(c) for c in combs])
        
        self.facility_combinations = all_combinations
        print(f"Generated {len(all_combinations)} facility combinations")
        return all_combinations
    
    def compute_refueling_frequency(self, origin: Any, destination: Any) -> float:
 
        od_pair = (origin, destination)
        
        if od_pair not in self.flow_paths:
            raise ValueError(f"No path exists for OD pair {od_pair}")
        
        path = self.flow_paths[od_pair]

        roundtrip_distance = 0
        for i in range(len(path) - 1):
            roundtrip_distance += self.network[path[i]][path[i+1]]['length']
        roundtrip_distance *= 2
        
        if roundtrip_distance <= 0:
            return 1.0

        trips_per_tank = max(1, int(self.vehicle_range / roundtrip_distance))
        e_q = 1.0 / trips_per_tank
        
        self.e_coefficients[od_pair] = e_q
        return e_q
    
    def compute_facility_usage(self, 
                              origin: Any, 
                              destination: Any, 
                              facility: Any, 
                              combination: List) -> int:
        """
        Compute facility usage coefficient for capacitated model. (0, 1, or 2)
        """
        if facility not in combination:
            return 0
        
        if facility == origin or facility == destination:
            return 1
        
        return 2
    
    def check_path_refueling_feasibility(self, path: List, facilities: List) -> bool:
        """
        Check if a path can be traversed with given facilities. True if path is feasible, False otherwise. 
        """
        if len(path) < 2:
            return False

        total_length = sum(self.network[path[i]][path[i+1]]['length'] 
                          for i in range(len(path) - 1))

        if total_length * 2 <= self.vehicle_range:
            return any(facility in path for facility in facilities)

        remaining_range = self.vehicle_range / 2
        if path[0] in facilities:
            remaining_range = self.vehicle_range
        
        for i in range(1, len(path)):
            segment_length = self.network[path[i-1]][path[i]]['length']
            remaining_range -= segment_length
            
            if remaining_range < 0:
                return False
            
            if path[i] in facilities:
                remaining_range = self.vehicle_range

        remaining_range = self.vehicle_range / 2
        if path[-1] in facilities:
            remaining_range = self.vehicle_range
        
        for i in range(len(path) - 1, 0, -1):
            segment_length = self.network[path[i-1]][path[i]]['length']
            remaining_range -= segment_length
            
            if remaining_range < 0:
                return False
            
            if path[i-1] in facilities:
                remaining_range = self.vehicle_range
        
        return True
    
    def generate_path_refueling_combinations(self) -> Dict:
        """
        Generate dictionary mapping OD pairs to valid facility combinations
        """
        if not self.facility_combinations:
            self.generate_facility_combinations()
        
        path_refueling_combinations = {}
        
        for od_pair, path in tqdm(self.flow_paths.items(), desc="Generating combinations"):
            valid_combinations = []
            
            for combo in self.facility_combinations:
                if self.check_path_refueling_feasibility(path, combo):
                    valid_combinations.append(combo)
            
            path_refueling_combinations[od_pair] = valid_combinations

            self.compute_refueling_frequency(od_pair[0], od_pair[1])
        
        self.path_refueling_combinations = path_refueling_combinations
        
        total_valid = sum(len(combinations) for combinations in path_refueling_combinations.values())
        print(f"Generated {total_valid} valid combinations across {len(path_refueling_combinations)} OD pairs")
        
        return path_refueling_combinations
    
    def calculate_node_weights(self, method: str ='double') -> np.ndarray:
        """
        str, default='double'
            Weight calculation method ('single' or 'double')
        """
        if not self.flows:
            raise ValueError("Flows must be loaded first")

        max_node = max(max(od[0], od[1]) for od in self.flows.keys())
        node_weights = np.zeros(max_node + 1)
        total_flow = sum(self.flows.values())
        
        if method == 'single':
            # Origin-only weights (paper definition)
            for (origin, destination), volume in self.flows.items():
                node_weights[origin] += volume
            node_weights = node_weights / total_flow
            
        elif method == 'double':
            # MOSEL-style weights (both origin and destination)
            for (origin, destination), volume in self.flows.items():
                node_weights[origin] += volume
                node_weights[destination] += volume
            node_weights = node_weights / (total_flow * 2)
            
        else:
            raise ValueError("Method must be 'single' or 'double'")
        
        self.node_weights = node_weights
        print(f"Calculated node weights using {method} method")
        return node_weights
    
    def solve(self, 
         solver: str = "greedy", 
         solver_instance: Optional[Any] = None,
         max_time_seconds: Optional[int] = 600,
         objective: str = "flow",
         weight_method: str = "double",
         seed: Optional[int] = None,
         initialization_method: str = "empty", 
         max_iterations: int = 100,
         **kwargs) -> Dict:
        """
        Solve the FRLM problem with specified solver.
        
        Parameters:
        -----------
        solver : str, default="greedy"
            Solver to use: "greedy", "pulp", "exact"
        solver_instance : Optional[Any], default=None
            Specific solver instance for PuLP (e.g., pulp.GUROBI_CMD())
        max_time_seconds : Optional[int], default=600
            Maximum solution time in seconds
        objective : str, default="flow"
            Objective type: "flow", "vmt" (for capacitated model)
        weight_method : str, default="double"
            Node weight calculation method for threshold extension
        **kwargs
            Additional solver-specific parameters

        """
        if not self.network:
            raise ValueError("No network loaded. ")
        
        if not self.flows:
            raise ValueError("No flows loaded. ")

        self._clear_solution_data()

        if isinstance(solver, str) and solver.lower() == "greedy":
            return self._solve_greedy(objective=objective, weight_method=weight_method,
                                    seed=seed, initialization_method=initialization_method,
                                    max_iterations=max_iterations, **kwargs)
        elif isinstance(solver, str) and solver.lower() == "exact":
            return self._solve_pulp(solver_instance=solver_instance, 
                                max_time_seconds=max_time_seconds,
                                objective=objective, weight_method=weight_method, **kwargs)
        elif solver_instance is None and hasattr(solver, 'solve'):  
            return self._solve_pulp(solver_instance=solver, 
                                max_time_seconds=max_time_seconds,
                                objective=objective, weight_method=weight_method, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'greedy' or pass PuLP solver instance")
    
    def _clear_solution_data(self):
        self.variables = {}
        self.constraints = {}
        self.lagrange_multipliers = {}
        self.reduced_costs = {}
        self.shadow_prices = {}
        self.greedy_iterations = []
        self.marginal_contributions = {}
        self.solver_stats = {}
        self.bounds = {'lower': None, 'upper': None}
        self.model = None
        self.pulp_status = None
    
    def _solve_greedy(self, 
                 objective: str = "flow",
                 weight_method: str = "double",
                 seed: Optional[int] = None,
                 initialization_method: str = "empty",
                 max_iterations: int = 100,
                 **kwargs) -> Dict:
        
        import random
        import numpy as np

        self.solver_type = "greedy"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if kwargs.get('verbose', True):
                print(f"Random seed set to: {seed}")

        start_time = time.time()
        
        if self.capacitated:
            return self._solve_capacitated_greedy(objective=objective, **kwargs)
        else:
            return self._solve_basic_greedy(weight_method=weight_method, **kwargs)
    
    def _initialize_greedy_solution(self, method: str) -> set:

        import random
        
        if method == "empty":
            #  start with NO pre-selected facilities 
            return set()
        
        elif method == "random":
            # Start with 1-2 random facilities
            init_count = random.randint(1, min(2, len(self.candidate_sites)))
            return set(random.sample(self.candidate_sites, init_count))
        
        elif method == "central":
            # Start with most central nodes (highest degree of connection)
            if hasattr(self.network, 'degree'):
                centrality = dict(self.network.degree())
                sorted_nodes = sorted(self.candidate_sites, 
                                    key=lambda x: centrality.get(x, 0), reverse=True)
                init_count = min(2, len(sorted_nodes))
                return set(sorted_nodes[:init_count])
            else:
                return set([self.candidate_sites[0]]) if self.candidate_sites else set()
        
        elif method == "high_flow":  
            # Start with nodes on paths with highest flow
            node_importance = {node: 0 for node in self.candidate_sites}
            
            for od_pair, volume in self.flows.items():
                if od_pair in self.flow_paths:
                    path = self.flow_paths[od_pair]
                    for node in path:
                        if node in node_importance:
                            node_importance[node] += volume
            
            sorted_nodes = sorted(self.candidate_sites, 
                                key=lambda x: node_importance[x], reverse=True)
            init_count = min(2, len(sorted_nodes))
            return set(sorted_nodes[:init_count])
        
        else:
            print(f"Warning: Unknown initialisation method '{method}', using 'empty'")
            return set()
    
    def _solve_basic_greedy(self, 
                       weight_method: str = "double",
                       seed: Optional[int] = None,
                       initialization_method: str = "empty", 
                       max_iterations: int = 100,
                       **kwargs) -> Dict:

        if not self.K or not self.a:
            self.generate_k_a_sets()

        import random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if kwargs.get('verbose', True):
                print(f"Random seed set to: {seed}")
        
        start_time = time.time()
        
        num_facilities = max([node for key in self.K for node in self.K[key]]) if self.K else 0
        num_flows = len(self.a)

        z = np.zeros(num_facilities + 1, dtype=int)  
        y = np.zeros(num_flows + 1, dtype=int)       
        c = None  # Covered nodes (for threshold extension)

        iterations = []
        marginal_contributions = {}

        if self.threshold > 0:
            self.calculate_node_weights(method=weight_method)
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
                            list(self.flows.values())[q-1] 
                            for q in origin_flows[origin] 
                            if y[q] == 1
                        )
                        if covered_flow >= self.threshold * origin_total_flow[origin]:
                            c[origin] = 1

        remaining_facilities = self.p - len(initial_facilities)
        iteration_count = 0
        
        for facility_count in range(remaining_facilities):
            iteration_count += 1

            if iteration_count > max_iterations:
                if kwargs.get('verbose', True):
                    print(f"Reached maximum iterations ({max_iterations}), stopping early")
                break
            
            best_facility = None
            best_value = 0
            iteration_candidates = {}
            
            for i in range(1, num_facilities + 1):
                if z[i] == 1:  #
                    continue

                z[i] = 1

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
                                list(self.flows.values())[q-1] 
                                for q in origin_flows[origin] 
                                if new_y[q] == 1
                            )
                            if covered_flow >= self.threshold * origin_total_flow[origin]:
                                new_c[origin] = 1
                    
                    weighted_coverage = sum(self.node_weights[j] * new_c[j] 
                                        for j in range(len(new_c)))
                    flow_coverage = sum(list(self.flows.values())[q-1] * new_y[q] 
                                    for q in range(1, len(new_y)) if q <= len(self.flows)) / sum(self.flows.values())
                    evaluation = self.weight * weighted_coverage + (1 - self.weight) * flow_coverage
                else:
                    evaluation = sum(list(self.flows.values())[q-1] 
                                for q in range(1, len(new_y)) 
                                if q <= len(self.flows) and new_y[q] == 1)
                
                z[i] = 0  

                marginal_benefit = evaluation - (self.objective_value if self.objective_value else 0)
                iteration_candidates[i] = {
                    'facility_id': i,
                    'objective_value': evaluation,
                    'marginal_benefit': marginal_benefit
                }
                marginal_contributions[i] = marginal_benefit
                
                if evaluation > best_value:
                    best_value = evaluation
                    best_facility = i

            iteration_info = {
                'iteration': len(initial_facilities) + facility_count + 1,  # Adjust for initial facilities
                'selected_facility': best_facility,
                'objective_value': best_value,
                'candidates_evaluated': iteration_candidates,
                'marginal_benefit': iteration_candidates.get(best_facility, {}).get('marginal_benefit', 0)
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
                                list(self.flows.values())[q-1] 
                                for q in origin_flows[origin] 
                                if y[q] == 1
                            )
                            if covered_flow >= self.threshold * origin_total_flow[origin]:
                                c[origin] = 1
        

        self.solution_time = time.time() - start_time
        self.status = "Heuristic" 
        self.greedy_iterations = iterations
        self.marginal_contributions = marginal_contributions

        #for i in range(1, num_facilities + 1):
            #self.variables[f"z_{i}"] = int(z[i])
        #for q in range(1, num_flows + 1):
            #if q in self.a:
                #self.variables[f"y_{q}"] = int(y[q])

        self.constraints = {}
        self.constraints["facility_count"] = {
            'name': "facility_count",
            'type': "eq",
            'rhs': float(self.p),
            'slack': 0.0,  
            'shadow_price': self._estimate_facility_shadow_price(),
            'active': True
        }


        for od_pair, coverage in self.flow_coverage.items():
            if coverage.get('covered_proportion', 0.0) > 0.99:  # Fully covered
                constraint_name = f"coverage_{od_pair[0]}_{od_pair[1]}"
                self.constraints[constraint_name] = {
                    'name': constraint_name,
                    'type': "geq", 
                    'rhs': 1.0,
                    'slack': 0.0,
                    'shadow_price': self.flows[od_pair],
                    'active': True
                }

        for site in self.candidate_sites:
            is_selected = site in self.selected_facilities
            self.variables[f"x_{site}"] = 1.0 if is_selected else 0.0

        for i, od_pair in enumerate(self.flows.keys()):
            if i + 1 <= len(y) and y[i + 1] == 1:
                self.variables[f"y_{od_pair[0]}_{od_pair[1]}"] = 1.0
            else:
                self.variables[f"y_{od_pair[0]}_{od_pair[1]}"] = 0.0
        
        self.selected_facilities = {i: 1 for i in range(1, num_facilities + 1) if z[i] == 1}

        covered_flows = [q for q in range(1, num_flows + 1) if q in self.a and y[q] == 1]
        total_flow_covered = sum(list(self.flows.values())[q-1] for q in covered_flows if q <= len(self.flows))
        coverage_percentage = total_flow_covered / sum(self.flows.values()) if sum(self.flows.values()) > 0 else 0
        
        if self.threshold > 0 and c is not None:
            self.covered_nodes = [j for j in range(len(c)) if c[j] == 1]
            weighted_coverage = sum(self.node_weights[j] * c[j] for j in range(len(c)))
            self.objective_value = self.weight * weighted_coverage + (1 - self.weight) * coverage_percentage
        else:
            self.covered_nodes = []
            self.objective_value = total_flow_covered

        self.flow_coverage = {}
        for i, od_pair in enumerate(self.flows.keys()):
            if i + 1 <= len(y) and y[i + 1] == 1:
                self.flow_coverage[od_pair] = {
                    'flow_volume': self.flows[od_pair],
                    'covered_proportion': 1.0,
                    'covered_volume': self.flows[od_pair]
                }
            else:
                self.flow_coverage[od_pair] = {
                    'flow_volume': self.flows[od_pair],
                    'covered_proportion': 0.0,
                    'covered_volume': 0.0
                }

        self.solver_stats = {
            'total_candidates_evaluated': sum(len(iter_info['candidates_evaluated']) for iter_info in iterations),
            'average_candidates_per_iteration': np.mean([len(iter_info['candidates_evaluated']) for iter_info in iterations]) if iterations else 0,
            'convergence_pattern': [iter_info['objective_value'] for iter_info in iterations],
            'marginal_benefits': [iter_info['marginal_benefit'] for iter_info in iterations],
            'initialization_method': initialization_method,
            'initial_facilities': list(initial_facilities),
            'seed_used': seed
        }
        

        if not hasattr(self, 'bounds'):
            self.bounds = {}
        self.bounds['upper'] = self.objective_value 
        self.bounds['lower'] = self.objective_value

        print(f"\nGreedy Basic FRLM Solution:")
        if initial_facilities:
            print(f"Initialized with {len(initial_facilities)} facilities using '{initialization_method}' method")
        print(f"Selected {len(self.selected_facilities)} facilities")
        print(f"Covered {len(covered_flows)} flows ({coverage_percentage:.2%} of total flow)")
        if self.threshold > 0:
            print(f"Covered {len(self.covered_nodes)} nodes")
            print(f"Objective value: {self.objective_value:.4f}")
        else:
            print(f"Total flow covered: {total_flow_covered:.2f}")
        print(f"Solution time: {self.solution_time:.2f} seconds")
        if iteration_count >= max_iterations:
            print(f"Stopped at iteration limit ({max_iterations})")
        
        return {
            'status': self.status,
            'objective_value': self.objective_value,
            'selected_facilities': self.selected_facilities,
            'flow_coverage': self.flow_coverage,
            'covered_nodes': self.covered_nodes,
            'total_flow': sum(self.flows.values()),
            'covered_flow': total_flow_covered,
            'coverage_percentage': coverage_percentage,
            'solution_time': self.solution_time,
            'initialization_method': initialization_method,
            'initial_facilities': list(initial_facilities),
            'seed_used': seed,
            'iterations_completed': len(iterations),
            'solver_stats': self.solver_stats
        }
    
    def _solve_capacitated_greedy(self, objective: str = "flow", **kwargs) -> Dict:

        if not self.path_refueling_combinations:
            self.generate_path_refueling_combinations()
        
        start_time = time.time()
        
        path_distances = {}
        if objective == 'vmt':
            for od_pair, path in self.flow_paths.items():
                distance = sum(self.network[path[i]][path[i+1]]['length'] 
                             for i in range(len(path) - 1))
                path_distances[od_pair] = distance
        
        current_facilities = {}
        current_objective = 0
        iterations = []
        marginal_contributions = {}

        for iteration in range(self.p):
            best_location = None
            best_improvement = 0
            iteration_candidates = {}
            
            for k in self.candidate_sites:
                candidate_facilities = current_facilities.copy()
                candidate_facilities[k] = candidate_facilities.get(k, 0) + 1

                evaluation = self._evaluate_capacitated_solution(candidate_facilities)
                
                if evaluation['status'] == 'Optimal':
                    if objective == 'flow':
                        obj_value = evaluation.get('objective_value', 0.0)
                        if obj_value is None:
                            obj_value = 0.0
                        improvement = obj_value - current_objective
                    elif objective == 'vmt':
                        if evaluation.get('flow_coverage'):
                            vmt_covered = sum(
                                evaluation['flow_coverage'][od_pair].get('covered_proportion', 0.0) * 
                                self.flows[od_pair] * 
                                path_distances[od_pair]
                                for od_pair in evaluation['flow_coverage']
                            )
                            improvement = vmt_covered - current_objective
                        else:
                            improvement = 0.0
                    
                    iteration_candidates[k] = {
                        'facility_id': k,
                        'objective_value': evaluation['objective_value'],
                        'marginal_benefit': improvement
                    }
                    marginal_contributions[k] = improvement
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_location = k

            iteration_info = {
                'iteration': iteration + 1,
                'selected_facility': best_location,
                'objective_value': current_objective + best_improvement,
                'candidates_evaluated': iteration_candidates,
                'marginal_benefit': best_improvement
            }
            iterations.append(iteration_info)
            
            if best_location is None:
                print(f"No further improvement possible after {iteration} iterations")
                break
            

            current_facilities[best_location] = current_facilities.get(best_location, 0) + 1
            current_objective += best_improvement
            
            print(f"Iteration {iteration+1}: Added module at {best_location}, objective: {current_objective:.2f}")

        self.selected_facilities = current_facilities
        self.greedy_iterations = iterations
        self.marginal_contributions = marginal_contributions
        
        final_evaluation = self._evaluate_capacitated_solution(current_facilities)
        self.flow_coverage = final_evaluation['flow_coverage']
        
        if objective == 'flow':
            self.objective_value = final_evaluation['objective_value']
        elif objective == 'vmt':
            self.objective_value = sum(
                final_evaluation['flow_coverage'][od_pair]['covered_proportion'] * 
                self.flows[od_pair] * 
                path_distances[od_pair]
                for od_pair in final_evaluation['flow_coverage']
            )
        
        self.solution_time = time.time() - start_time
        self.status = "Heuristic"

        self.solver_stats = {
            'total_candidates_evaluated': sum(len(iter_info['candidates_evaluated']) for iter_info in iterations),
            'average_candidates_per_iteration': np.mean([len(iter_info['candidates_evaluated']) for iter_info in iterations]),
            'convergence_pattern': [iter_info['objective_value'] for iter_info in iterations],
            'marginal_benefits': [iter_info['marginal_benefit'] for iter_info in iterations]
        }
        
        print(f"\nGreedy Capacitated FRLM Solution:")
        print(f"Objective value ({objective}): {self.objective_value:.2f}")
        print(f"Total flow covered: {final_evaluation['covered_flow']:.2f} / {final_evaluation['total_flow']:.2f} ({final_evaluation['coverage_percentage']:.2%})")
        print(f"Modules located at {len(self.selected_facilities)} locations:")
        for k, count in self.selected_facilities.items():
            print(f"  - Location {k}: {count} module(s)")
        print(f"Solution time: {self.solution_time:.2f} seconds")
        
        result = final_evaluation.copy()
        result.update({
            'objective_value': self.objective_value,
            'solution_time': self.solution_time,
            'status': self.status,
            'objective_type': objective
        })
        
        if objective == 'vmt':
            total_vmt = sum(self.flows[od_pair] * path_distances[od_pair] for od_pair in self.flows)
            covered_vmt = self.objective_value
            vmt_coverage_percentage = covered_vmt / total_vmt if total_vmt > 0 else 0
            
            result.update({
                'total_vmt': total_vmt,
                'covered_vmt': covered_vmt,
                'vmt_coverage_percentage': vmt_coverage_percentage
            })
        
        return result
    
    def _solve_pulp(self, 
                   solver_instance: Optional[Any] = None,
                   max_time_seconds: Optional[int] = 600,
                   objective: str = "flow",
                   weight_method: str = "double",
                   **kwargs) -> Dict:

        start_time = time.time()
        self.solver_type = "pulp"
        
        if self.capacitated:
            return self._solve_capacitated_pulp(solver_instance, max_time_seconds, objective, **kwargs)
        else:
            return self._solve_basic_pulp(solver_instance, max_time_seconds, weight_method, **kwargs)
    
    def _solve_basic_pulp(self, 
                         solver_instance: Optional[Any] = None,
                         max_time_seconds: Optional[int] = 600,
                         weight_method: str = "double",
                         **kwargs) -> Dict:

        if not self.K or not self.a:
            self.generate_k_a_sets()
        
        start_time = time.time()
        model = pulp.LpProblem("Basic_FRLM", pulp.LpMaximize)
        self.model = model
        
        num_facilities = max([node for key in self.K for node in self.K[key]]) if self.K else 0
        num_flows = len(self.a)

        z = {i: pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) 
             for i in range(1, num_facilities + 1)}
        y = {q: pulp.LpVariable(f"y_{q}", cat=pulp.LpBinary) 
             for q in range(1, num_flows + 1) if q in self.a}

        constraint_refs = {}

        if self.threshold > 0:
            self.calculate_node_weights(method=weight_method)
            c = {j: pulp.LpVariable(f"c_{j}", cat=pulp.LpBinary) 
                 for j in range(1, len(self.node_weights)) if self.node_weights[j] > 0}

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
                    covered_flow = pulp.lpSum(self.flows[list(self.flows.keys())[q-1]] * y[q] 
                                            for q in flows_for_node if q in y)
                    constraint = covered_flow >= self.threshold * flow_sum * c[j]
                    model += constraint
                    constraint_refs[f"threshold_node_{j}"] = constraint

            model += (self.weight * pulp.lpSum(self.node_weights[j] * c[j] for j in c) + 
                     (1 - self.weight) * pulp.lpSum(list(self.flows.values())[q-1] * y[q] 
                                                   for q in y if q <= len(self.flows)) / sum(self.flows.values()))
        else:

            model += pulp.lpSum(list(self.flows.values())[q-1] * y[q] 
                              for q in y if q <= len(self.flows))

        for q in y:
            if q in self.a:
                for arc in self.a[q]:
                    key = (q, arc)
                    if key in self.K:
                        refuel_facilities = pulp.lpSum(z[node] for node in self.K[key] if node in z)
                        constraint = refuel_facilities >= y[q]
                        model += constraint
                        constraint_refs[f"flow_coverage_{q}_{arc}"] = constraint
        

        facility_constraint = pulp.lpSum(z.values()) == self.p
        model += facility_constraint
        constraint_refs["facility_count"] = facility_constraint

        if solver_instance is None:
            if max_time_seconds:
                solver_instance = pulp.GUROBI_CMD(timeLimit=max_time_seconds, msg=True)
            else:
                solver_instance = pulp.GUROBI_CMD(msg=True)
        
        model.solve(solver_instance)
        
        self.solution_time = time.time() - start_time
        self.pulp_status = model.status
        self.status = pulp.LpStatus[model.status]
        
        if model.status == pulp.LpStatusOptimal:
            self.objective_value = pulp.value(model.objective)

            for var in model.variables():
                self.variables[var.name] = var.varValue

            try:
                for constraint_name, constraint in constraint_refs.items():
                    if hasattr(constraint, 'pi') and constraint.pi is not None:
                        self.lagrange_multipliers[constraint_name] = constraint.pi
                        self.shadow_prices[constraint_name] = constraint.pi  # Alias
            except AttributeError:
                print("Warning: Dual values not available with current solver")

            try:
                for var in model.variables():
                    if hasattr(var, 'dj') and var.dj is not None:
                        self.reduced_costs[var.name] = var.dj
            except AttributeError:
                print("Warning: Reduced costs not available with current solver")
            
            self.selected_facilities = {i: 1 for i in z if pulp.value(z[i]) > 0.5}

            covered_flows = [q for q in y if pulp.value(y[q]) > 0.5]
            total_flow_covered = sum(list(self.flows.values())[q-1] 
                                   for q in covered_flows if q <= len(self.flows))
            coverage_percentage = total_flow_covered / sum(self.flows.values()) if sum(self.flows.values()) > 0 else 0
            
            self.flow_coverage = {}
            for i, od_pair in enumerate(self.flows.keys()):
                if i + 1 in y and pulp.value(y[i + 1]) > 0.5:
                    self.flow_coverage[od_pair] = {
                        'flow_volume': self.flows[od_pair],
                        'covered_proportion': 1.0,
                        'covered_volume': self.flows[od_pair]
                    }
                else:
                    self.flow_coverage[od_pair] = {
                        'flow_volume': self.flows[od_pair],
                        'covered_proportion': 0.0,
                        'covered_volume': 0.0
                    }
            
            if self.threshold > 0:
                self.covered_nodes = [j for j in c if pulp.value(c[j]) > 0.5]
            else:
                self.covered_nodes = []
            
            self.solver_stats = {
                'solver_name': str(solver_instance),
                'num_variables': len(model.variables()),
                'num_constraints': len([constraint for constraint in model.constraints.values()]),
                'objective_sense': 'maximize'
            }
            
            print(f"\nPuLP Basic FRLM Solution:")
            print(f"Status: {self.status}")
            print(f"Objective value: {self.objective_value:.4f}")
            print(f"Selected {len(self.selected_facilities)} facilities")
            print(f"Covered {len(covered_flows)} flows ({coverage_percentage:.2%} of total flow)")
            if self.threshold > 0:
                print(f"Covered {len(self.covered_nodes)} nodes")
            print(f"Solution time: {self.solution_time:.2f} seconds")
            
            return {
                'status': self.status,
                'objective_value': self.objective_value,
                'selected_facilities': self.selected_facilities,
                'flow_coverage': self.flow_coverage,
                'covered_nodes': self.covered_nodes,
                'total_flow': sum(self.flows.values()),
                'covered_flow': total_flow_covered,
                'coverage_percentage': coverage_percentage,
                'solution_time': self.solution_time
            }
        else:
            print(f"No optimal solution found. Status: {self.status}")
            return {'status': self.status}
    
    def _solve_capacitated_pulp(self, 
                               solver_instance: Optional[Any] = None,
                               max_time_seconds: Optional[int] = 600,
                               objective: str = "flow",
                               **kwargs) -> Dict:

        if not self.path_refueling_combinations:
            self.generate_path_refueling_combinations()
        
        start_time = time.time()

        path_distances = {}
        if objective == 'vmt':
            for od_pair, path in self.flow_paths.items():
                distance = sum(self.network[path[i]][path[i+1]]['length'] 
                             for i in range(len(path) - 1))
                path_distances[od_pair] = distance
        
        model_name = "Capacitated_FRLM_VMT" if objective == 'vmt' else "Capacitated_FRLM"
        model = pulp.LpProblem(model_name, pulp.LpMaximize)
        self.model = model

        x = {k: pulp.LpVariable(f"x_{k}", lowBound=0, cat=pulp.LpInteger) 
             for k in self.candidate_sites}
        
        y = {}
        for q, od_pair in enumerate(self.flows.keys()):
            valid_combinations = self.path_refueling_combinations[od_pair]
            for h, combination in enumerate(valid_combinations):
                y[(q, h)] = pulp.LpVariable(f"y_{q}_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        constraint_refs = {}

        if objective == 'vmt':
            model += pulp.lpSum(
                self.flows[od_pair] * path_distances[od_pair] * y[(q, h)]
                for q, od_pair in enumerate(self.flows.keys())
                for h, combination in enumerate(self.path_refueling_combinations[od_pair])
            )
        else:
            model += pulp.lpSum(
                self.flows[od_pair] * y[(q, h)]
                for q, od_pair in enumerate(self.flows.keys())
                for h, combination in enumerate(self.path_refueling_combinations[od_pair])
            )

        for k in self.candidate_sites:
            capacity_constraint = pulp.lpSum(
                self.e_coefficients[od_pair] * 
                self.compute_facility_usage(od_pair[0], od_pair[1], k, combination) * 
                self.flows[od_pair] * y[(q, h)]
                for q, od_pair in enumerate(self.flows.keys())
                for h, combination in enumerate(self.path_refueling_combinations[od_pair])
                if k in combination
            ) <= self.capacity * x[k]
            model += capacity_constraint
            constraint_refs[f"capacity_{k}"] = capacity_constraint

        module_constraint = pulp.lpSum(x.values()) == self.p
        model += module_constraint
        constraint_refs["module_count"] = module_constraint

        for q, od_pair in enumerate(self.flows.keys()):
            flow_constraint = pulp.lpSum(
                y[(q, h)] 
                for h in range(len(self.path_refueling_combinations[od_pair]))
            ) <= 1
            model += flow_constraint
            constraint_refs[f"flow_coverage_{q}"] = flow_constraint

        if solver_instance is None:
            if max_time_seconds:
                solver_instance = pulp.GUROBI_CMD(
                    timeLimit=max_time_seconds,
                    msg=True,
                    options=[("IntFeasTol", "1e-9"), ("MIPGap", "1e-9")]
                )
            else:
                solver_instance = pulp.GUROBI_CMD(
                    msg=True,
                    options=[("IntFeasTol", "1e-9"), ("MIPGap", "1e-9")]
                )
        
        model.solve(solver_instance)
        
        self.solution_time = time.time() - start_time
        self.pulp_status = model.status
        self.status = pulp.LpStatus[model.status]
        
        if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusNotSolved:
            self.objective_value = pulp.value(model.objective)

            for var in model.variables():
                self.variables[var.name] = var.varValue
            
            self.constraints = {}
            for constraint_name, constraint in constraint_refs.items():
                slack_val = getattr(constraint, 'slack', 0.0) if hasattr(constraint, 'slack') else 0.0
                shadow_val = getattr(constraint, 'pi', 0.0) if hasattr(constraint, 'pi') else 0.0
                
                self.constraints[constraint_name] = {
                    'name': constraint_name,
                    'type': "eq" if "=" in str(constraint) else "leq" if "<=" in str(constraint) else "geq",
                    'rhs': float(self.p), 
                    'slack': slack_val,
                    'shadow_price': shadow_val,
                    'active': abs(slack_val) < 1e-6
                }

            try:
                for constraint_name, constraint in constraint_refs.items():
                    if hasattr(constraint, 'pi') and constraint.pi is not None:
                        self.lagrange_multipliers[constraint_name] = constraint.pi
                        self.shadow_prices[constraint_name] = constraint.pi  # Alias
            except AttributeError:
                print("Warning: Dual values not available with current solver")
            
            try:
                for var in model.variables():
                    if hasattr(var, 'dj') and var.dj is not None:
                        self.reduced_costs[var.name] = var.dj
            except AttributeError:
                print("Warning: Reduced costs not available with current solver")
            
            self.selected_facilities = {k: int(x[k].value()) for k in self.candidate_sites if x[k].value() > 0}

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
                    'flow_volume': flow_volume,
                    'covered_proportion': covered_proportion,
                    'covered_volume': flow_volume * covered_proportion
                }
                
                if objective == 'vmt':
                    self.flow_coverage[od_pair]['distance'] = path_distances[od_pair]
                    self.flow_coverage[od_pair]['covered_vmt'] = flow_volume * path_distances[od_pair] * covered_proportion
                
                covered_flow += flow_volume * covered_proportion
            
            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0
            
            self.solver_stats = {
                'solver_name': str(solver_instance),
                'num_variables': len(model.variables()),
                'num_constraints': len([constraint for constraint in model.constraints.values()]),
                'objective_sense': 'maximize'
            }
            
            print(f"\nPuLP Capacitated FRLM Solution:")
            print(f"Status: {self.status}")
            print(f"Objective value ({objective}): {self.objective_value:.2f}")
            print(f"Total flow covered: {covered_flow:.2f} / {total_flow:.2f} ({coverage_percentage:.2%})")
            print(f"Modules located at {len(self.selected_facilities)} locations:")
            for k, count in self.selected_facilities.items():
                print(f"  - Location {k}: {count} module(s)")
            print(f"Solution time: {self.solution_time:.2f} seconds")
            
            result = {
                'status': self.status,
                'objective_value': self.objective_value,
                'selected_facilities': self.selected_facilities,
                'flow_coverage': self.flow_coverage,
                'total_flow': total_flow,
                'covered_flow': covered_flow,
                'coverage_percentage': coverage_percentage,
                'solution_time': self.solution_time,
                'objective_type': objective
            }
            
            if objective == 'vmt':
                total_vmt = sum(self.flows[od_pair] * path_distances[od_pair] for od_pair in self.flows)
                covered_vmt = sum(coverage.get('covered_vmt', 0) for coverage in self.flow_coverage.values())
                vmt_coverage_percentage = covered_vmt / total_vmt if total_vmt > 0 else 0
                
                result.update({
                    'total_vmt': total_vmt,
                    'covered_vmt': covered_vmt,
                    'vmt_coverage_percentage': vmt_coverage_percentage
                })
            
            return result
        else:
            print(f"No optimal solution found. Status: {self.status}")
            return {'status': self.status}
    
    def _evaluate_capacitated_solution(self, facilities: Dict[Any, int]) -> Dict:

        model = pulp.LpProblem("Evaluate_CFRLM", pulp.LpMaximize)
        
        y = {}
        for q, od_pair in enumerate(self.flows.keys()):
            valid_combinations = self.path_refueling_combinations[od_pair]
            for h, combination in enumerate(valid_combinations):
                if all(facility in facilities for facility in combination):
                    y[(q, h)] = pulp.LpVariable(f"y_{q}_{h}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
        

        model += pulp.lpSum(
            self.flows[od_pair] * y[(q, h)]
            for q, od_pair in enumerate(self.flows.keys())
            for h, combination in enumerate(self.path_refueling_combinations[od_pair])
            if (q, h) in y
        )
        

        for k in facilities.keys():
            model += pulp.lpSum(
                self.e_coefficients[od_pair] * 
                self.compute_facility_usage(od_pair[0], od_pair[1], k, combination) * 
                self.flows[od_pair] * y[(q, h)]
                for q, od_pair in enumerate(self.flows.keys())
                for h, combination in enumerate(self.path_refueling_combinations[od_pair])
                if (q, h) in y and k in combination
            ) <= self.capacity * facilities[k]
        

        for q, od_pair in enumerate(self.flows.keys()):
            model += pulp.lpSum(
                y[(q, h)] 
                for h in range(len(self.path_refueling_combinations[od_pair]))
                if (q, h) in y
            ) <= 1
        
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
                    'flow_volume': flow_volume,
                    'covered_proportion': covered_proportion,
                    'covered_volume': flow_volume * covered_proportion
                }
                
                covered_flow += flow_volume * covered_proportion
            
            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0
            
            return {
                'status': 'Optimal',
                'objective_value': objective_value,
                'flow_coverage': flow_coverage,
                'total_flow': total_flow,
                'covered_flow': covered_flow,
                'coverage_percentage': coverage_percentage
            }
        else:
            return {
                        'status': pulp.LpStatus[model.status],
                        'objective_value': 0.0,  
                        'flow_coverage': {},
                        'total_flow': sum(self.flows.values()),
                        'covered_flow': 0.0,
                        'coverage_percentage': 0.0
                    }

    def visualize_solution(self, 
                          output_file: Optional[str] = None, 
                          show_flows: bool = True, 
                          node_size: int = 300,
                          facility_size: int = 600,
                          node_color: str = 'lightblue',
                          facility_color: str = 'red') -> None:

        try:
            import matplotlib.pyplot as plt
            
            if not self.network:
                raise ValueError("No network loaded. Call load_network() first.")
            
            if not self.selected_facilities:
                raise ValueError("No solution available. Call solve() first.")

            pos = nx.get_node_attributes(self.network, 'pos')
            if not pos:
                print("No position attributes found. Using spring layout.")
                pos = nx.spring_layout(self.network, seed=42)
            
            plt.figure(figsize=(12, 10))

            nx.draw_networkx_edges(self.network, pos, alpha=0.5, width=1.0)
        
            regular_nodes = [n for n in self.network.nodes() if n not in self.selected_facilities]
            nx.draw_networkx_nodes(self.network, pos, nodelist=regular_nodes, 
                                  node_size=node_size, node_color=node_color)

            for facility, count in self.selected_facilities.items():
                size = facility_size * (count**0.7 if self.capacitated else 1)
                nx.draw_networkx_nodes(self.network, pos, nodelist=[facility], 
                                      node_size=size, node_color=facility_color)

            nx.draw_networkx_labels(self.network, pos, font_size=10)

            if show_flows and self.flow_coverage:
                for od_pair, coverage in self.flow_coverage.items():
                    if coverage['covered_proportion'] > 0:
                        path = self.flow_paths[od_pair]
                        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                        
                        line_width = 1 + 3 * (coverage['covered_volume'] / max(self.flows.values()))
                        
                        nx.draw_networkx_edges(self.network, pos, edgelist=path_edges, 
                                              width=line_width, alpha=0.7, edge_color='green')

            total_modules = sum(self.selected_facilities.values()) if self.capacitated else len(self.selected_facilities)
            model_type = "Capacitated" if self.capacitated else "Basic"
            plt.title(f"{model_type} FRLM Solution: {total_modules} modules at {len(self.selected_facilities)} locations")

            total_flow = sum(self.flows.values())
            covered_flow = sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values())
            coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0
            
            stats_text = f"Flow coverage: {coverage_percentage:.1%}\nObjective value: {self.objective_value:.1f}"
            
            if hasattr(self, 'covered_nodes') and self.covered_nodes:
                stats_text += f"\nCovered nodes: {len(self.covered_nodes)}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.7))
            
            plt.axis('off')
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                print(f"Visualisation saved to {output_file}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib is required.")
    
    def print_solution_summary(self) -> None:

        if not self.selected_facilities:
            print("No solution available. Call solve() first.")
            return
        
        print("\n" + "="*60)
        print("FLOW REFUELING LOCATION MODEL - SOLUTION SUMMARY")
        print("="*60)
        
        model_type = "Capacitated" if self.capacitated else "Basic"
        print(f"Model Type: {model_type}")
        print(f"Solver Type: {self.solver_type}")
        print(f"Vehicle Range: {self.vehicle_range}")
        print(f"Facilities to Locate: {self.p}")
        if self.capacitated:
            print(f"Facility Capacity: {self.capacity}")
        if self.threshold > 0:
            print(f"Coverage Threshold: {self.threshold}")
            print(f"Weight Parameter: {self.weight}")
        
        print(f"\nSolution Status: {self.status}")
        print(f"Solution Time: {self.solution_time:.2f} seconds")
        print(f"Objective Value: {self.objective_value:.4f}")

        total_modules = sum(self.selected_facilities.values()) if self.capacitated else len(self.selected_facilities)
        print(f"\nFacility Locations ({total_modules} modules at {len(self.selected_facilities)} sites):")
        for facility, count in sorted(self.selected_facilities.items()):
            if self.capacitated:
                print(f"  - Site {facility}: {count} module(s)")
            else:
                print(f"  - Site {facility}")

        total_flow = sum(self.flows.values())
        covered_flow = sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values())
        coverage_percentage = covered_flow / total_flow if total_flow > 0 else 0
        
        print(f"\nFlow Coverage:")
        print(f"  - Total Flow: {total_flow:.2f}")
        print(f"  - Covered Flow: {covered_flow:.2f}")
        print(f"  - Coverage Percentage: {coverage_percentage:.2%}")

        fully_covered = sum(1 for coverage in self.flow_coverage.values() 
                           if coverage['covered_proportion'] >= 0.99)
        partially_covered = sum(1 for coverage in self.flow_coverage.values() 
                               if 0 < coverage['covered_proportion'] < 0.99)
        uncovered = sum(1 for coverage in self.flow_coverage.values() 
                       if coverage['covered_proportion'] <= 0.01)
        
        print(f"  - Fully Covered Flows: {fully_covered} / {len(self.flows)}")
        print(f"  - Partially Covered Flows: {partially_covered} / {len(self.flows)}")
        print(f"  - Uncovered Flows: {uncovered} / {len(self.flows)}")

        if hasattr(self, 'covered_nodes') and self.covered_nodes:
            total_origins = len(set(od[0] for od in self.flows.keys()))
            print(f"\nNode Coverage:")
            print(f"  - Covered Origin Nodes: {len(self.covered_nodes)} / {total_origins}")

        print(f"\nSolver Information:")
        if self.solver_type == "greedy":
            print(f"  - Greedy Iterations: {len(self.greedy_iterations)}")
            if self.solver_stats:
                print(f"  - Total Candidates Evaluated: {self.solver_stats.get('total_candidates_evaluated', 'N/A')}")
                print(f"  - Avg Candidates per Iteration: {self.solver_stats.get('average_candidates_per_iteration', 'N/A'):.1f}")

            if self.marginal_contributions:
                print(f"  - Marginal Benefits Available: {len(self.marginal_contributions)} facilities")
        
        elif self.solver_type == "pulp":
            if self.solver_stats:
                print(f"  - Number of Variables: {self.solver_stats.get('num_variables', 'N/A')}")
                print(f"  - Number of Constraints: {self.solver_stats.get('num_constraints', 'N/A')}")

            if self.lagrange_multipliers:
                print(f"  - Lagrange Multipliers Available: {len(self.lagrange_multipliers)} constraints")
            if self.reduced_costs:
                print(f"  - Reduced Costs Available: {len(self.reduced_costs)} variables")
        
        print("="*60)
    
    def print_solver_details(self) -> None:

        if not self.solver_type:
            print("No solver results available. Call solve() first.")
            return
        
        print("\n" + "="*60)
        print("DETAILED SOLVER INFORMATION")
        print("="*60)
        
        if self.solver_type == "greedy":
            print("GREEDY SOLVER DETAILS:")
            print(f"Total Iterations: {len(self.greedy_iterations)}")
            
            if self.greedy_iterations:
                print("\nIteration Details:")
                for i, iteration in enumerate(self.greedy_iterations):
                    print(f"  Iteration {iteration['iteration']}:")
                    print(f"    - Selected Facility: {iteration['selected_facility']}")
                    print(f"    - Objective Value: {iteration['objective_value']:.4f}")
                    print(f"    - Marginal Benefit: {iteration['marginal_benefit']:.4f}")
                    print(f"    - Candidates Evaluated: {len(iteration['candidates_evaluated'])}")
            
            if self.marginal_contributions:
                print(f"\nMarginal Contributions:")
                sorted_contributions = sorted(self.marginal_contributions.items(), 
                                           key=lambda x: x[1], reverse=True)
                for facility, contribution in sorted_contributions[:10]:  # Top 10
                    print(f"  Facility {facility}: {contribution:.4f}")
        
        elif self.solver_type == "pulp":
            print("PULP SOLVER DETAILS:")
            print(f"PuLP Status: {self.pulp_status}")
            print(f"Model: {self.model.name if self.model else 'N/A'}")
            
            if self.solver_stats:
                print(f"Variables: {self.solver_stats.get('num_variables', 'N/A')}")
                print(f"Constraints: {self.solver_stats.get('num_constraints', 'N/A')}")
                print(f"Objective Sense: {self.solver_stats.get('objective_sense', 'N/A')}")

            if self.lagrange_multipliers:
                print(f"\nLagrange Multipliers (sample):")
                sample_duals = list(self.lagrange_multipliers.items())[:5]
                for constraint_name, dual_value in sample_duals:
                    print(f"  {constraint_name}: {dual_value:.6f}")
                if len(self.lagrange_multipliers) > 5:
                    print(f"  ... and {len(self.lagrange_multipliers) - 5} more")

            if self.reduced_costs:
                print(f"\nReduced Costs (sample):")
                sample_costs = list(self.reduced_costs.items())[:5]
                for var_name, reduced_cost in sample_costs:
                    print(f"  {var_name}: {reduced_cost:.6f}")
                if len(self.reduced_costs) > 5:
                    print(f"  ... and {len(self.reduced_costs) - 5} more")
        
        print("="*60)
    
    def export_solution(self, filename: str, format: str = 'csv') -> None:

        if not self.selected_facilities:
            print("No solution available. Call solve() first.")
            return
        
        if format.lower() == 'csv':
            facilities_df = pd.DataFrame([
                {'facility_id': k, 'modules': v} 
                for k, v in self.selected_facilities.items()
            ])
            facilities_df.to_csv(filename.replace('.csv', '_facilities.csv'), index=False)
            coverage_data = []
            for od_pair, coverage in self.flow_coverage.items():
                coverage_data.append({
                    'origin': od_pair[0],
                    'destination': od_pair[1],
                    'flow_volume': coverage['flow_volume'],
                    'covered_proportion': coverage['covered_proportion'],
                    'covered_volume': coverage['covered_volume']
                })
            coverage_df = pd.DataFrame(coverage_data)
            coverage_df.to_csv(filename.replace('.csv', '_coverage.csv'), index=False)

            if self.solver_type == "greedy" and self.greedy_iterations:
                iterations_data = []
                for iteration in self.greedy_iterations:
                    iterations_data.append({
                        'iteration': iteration['iteration'],
                        'selected_facility': iteration['selected_facility'],
                        'objective_value': iteration['objective_value'],
                        'marginal_benefit': iteration['marginal_benefit'],
                        'candidates_evaluated': len(iteration['candidates_evaluated'])
                    })
                iterations_df = pd.DataFrame(iterations_data)
                iterations_df.to_csv(filename.replace('.csv', '_greedy_iterations.csv'), index=False)
            
            elif self.solver_type == "pulp" and self.lagrange_multipliers:
                duals_data = []
                for constraint_name, dual_value in self.lagrange_multipliers.items():
                    duals_data.append({
                        'constraint_name': constraint_name,
                        'lagrange_multiplier': dual_value
                    })
                duals_df = pd.DataFrame(duals_data)
                duals_df.to_csv(filename.replace('.csv', '_lagrange_multipliers.csv'), index=False)
            
            print(f"Solution exported to CSV files with base name: {filename}")
        
        elif format.lower() == 'json':
            import json
            
            solution_data = {
                'model_type': 'capacitated' if self.capacitated else 'basic',
                'solver_type': self.solver_type,
                'parameters': {
                    'vehicle_range': self.vehicle_range,
                    'p': self.p,
                    'capacity': self.capacity if self.capacitated else None,
                    'threshold': self.threshold,
                    'weight': self.weight
                },
                'solution': {
                    'status': self.status,
                    'objective_value': self.objective_value,
                    'solution_time': self.solution_time,
                    'selected_facilities': self.selected_facilities,
                    'flow_coverage': {str(k): v for k, v in self.flow_coverage.items()},
                    'covered_nodes': self.covered_nodes if hasattr(self, 'covered_nodes') else []
                },
                'solver_details': {
                    'greedy_iterations': self.greedy_iterations if self.solver_type == "greedy" else None,
                    'marginal_contributions': self.marginal_contributions if self.solver_type == "greedy" else None,
                    'lagrange_multipliers': self.lagrange_multipliers if self.solver_type == "pulp" else None,
                    'reduced_costs': self.reduced_costs if self.solver_type == "pulp" else None,
                    'solver_stats': self.solver_stats
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(solution_data, f, indent=2)
            
            print(f"Solution exported to {filename}")
        
        elif format.lower() == 'excel':
            with pd.ExcelWriter(filename) as writer:
                facilities_df = pd.DataFrame([
                    {'facility_id': k, 'modules': v} 
                    for k, v in self.selected_facilities.items()
                ])
                facilities_df.to_excel(writer, sheet_name='Facilities', index=False)

                coverage_data = []
                for od_pair, coverage in self.flow_coverage.items():
                    coverage_data.append({
                        'origin': od_pair[0],
                        'destination': od_pair[1],
                        'flow_volume': coverage['flow_volume'],
                        'covered_proportion': coverage['covered_proportion'],
                        'covered_volume': coverage['covered_volume']
                    })
                coverage_df = pd.DataFrame(coverage_data)
                coverage_df.to_excel(writer, sheet_name='Flow_Coverage', index=False)

                summary_data = {
                    'Metric': ['Model Type', 'Solver Type', 'Vehicle Range', 'Facilities to Locate', 
                              'Facility Capacity', 'Solution Status', 'Objective Value',
                              'Solution Time (s)', 'Total Flow', 'Covered Flow', 'Coverage %'],
                    'Value': [
                        'Capacitated' if self.capacitated else 'Basic',
                        self.solver_type,
                        self.vehicle_range,
                        self.p,
                        self.capacity if self.capacitated else 'N/A',
                        self.status,
                        self.objective_value,
                        self.solution_time,
                        sum(self.flows.values()),
                        sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values()),
                        f"{sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values()) / sum(self.flows.values()) * 100:.2f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                if self.solver_type == "greedy" and self.greedy_iterations:
                    iterations_data = []
                    for iteration in self.greedy_iterations:
                        iterations_data.append({
                            'iteration': iteration['iteration'],
                            'selected_facility': iteration['selected_facility'],
                            'objective_value': iteration['objective_value'],
                            'marginal_benefit': iteration['marginal_benefit'],
                            'candidates_evaluated': len(iteration['candidates_evaluated'])
                        })
                    iterations_df = pd.DataFrame(iterations_data)
                    iterations_df.to_excel(writer, sheet_name='Greedy_Iterations', index=False)
                
                elif self.solver_type == "pulp":
                    if self.lagrange_multipliers:
                        duals_data = []
                        for constraint_name, dual_value in self.lagrange_multipliers.items():
                            duals_data.append({
                                'constraint_name': constraint_name,
                                'lagrange_multiplier': dual_value
                            })
                        duals_df = pd.DataFrame(duals_data)
                        duals_df.to_excel(writer, sheet_name='Lagrange_Multipliers', index=False)
                    
                    if self.reduced_costs:
                        costs_data = []
                        for var_name, reduced_cost in self.reduced_costs.items():
                            costs_data.append({
                                'variable_name': var_name,
                                'reduced_cost': reduced_cost
                            })
                        costs_df = pd.DataFrame(costs_data)
                        costs_df.to_excel(writer, sheet_name='Reduced_Costs', index=False)
            
            print(f"Solution exported to {filename}")
        
        else:
            raise ValueError("Format must be 'csv', 'json', or 'excel'")
    
    def get_shadow_prices(self) -> Dict[str, float]:
        if self.solver_type == "pulp":
            return self.shadow_prices.copy()  
        elif self.solver_type == "greedy":
            shadow_prices = {}

            shadow_prices["facility_count"] = self._estimate_facility_shadow_price()
            
            for od_pair, coverage in self.flow_coverage.items():
                if coverage.get('covered_proportion', 0.0) > 0.99: 
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
                if var_name.startswith(('x_', 'z_')) and value < 0.5:  # Unselected facilities
                    if hasattr(self, 'marginal_contributions'):
                        node_id = int(var_name.split('_')[1]) if '_' in var_name else 0
                        reduced_costs[var_name] = -self.marginal_contributions.get(node_id, 0.0)
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
                    if coverage.get('covered_proportion', 0.0) < 1.0:
                        path = self.flow_paths[od_pair]
                        if candidate in path:
                            uncovered = self.flows[od_pair] * (1.0 - coverage.get('covered_proportion', 0.0))
                            improvement += uncovered
                
                best_improvement = max(best_improvement, improvement)
        
        return best_improvement

    @property
    def pulp(self):
        return self.problem

    def print_shadow_prices(self):
        print(f"\nSHADOW PRICES:")
        print(f"Solver: {self.solver_type}")
        
        shadow_prices = self.get_shadow_prices()
        if shadow_prices:
            for constraint_name, price in shadow_prices.items():
                if abs(price) > 1e-6:  # Only show non-zero shadow prices
                    print(f"  {constraint_name}: π = {price:.4f}")
        else:
            print("No shadow price information available")
            
        print(f"Total constraints with shadow prices: {len(shadow_prices)}")
        print(f"Total variables: {len(self.variables)}")

    
    @property
    def problem(self):
        if self.solver_type == "pulp":
            return self.model 
        elif self.solver_type == "greedy":
            return self._create_greedy_problem_view()
        return None
    
    def _create_greedy_problem_view(self):
        class GreedyProblemView:
            def __init__(self, frlm_instance):
                self.frlm = frlm_instance
                self.name = f"Greedy_{frlm_instance.__class__.__name__}"
                self.status = 1 if frlm_instance.status == "Heuristic" else 0
            
            @property
            def variables(self):
                return [GreedyVariable(name, value) for name, value in self.frlm.variables.items()]
            
            @property
            def constraints(self):
                return {name: GreedyConstraint(name, info) 
                       for name, info in self.frlm.constraints.items()}
            
            def objective_value(self):
                return self.frlm.objective_value
        
        return GreedyProblemView(self)
    
    def get_solution_details(self) -> Dict:
        return {
            'solver_type': self.solver_type,
            'model_type': 'capacitated' if self.capacitated else 'basic',
            'problem_instance': self.problem,
            'optimisation_status': self.status,
            'objective_value': self.objective_value,
            'solution_time': self.solution_time,
            'facilities': self._get_facility_details(),
            'flows': self._get_flow_details(),
            'constraints': self._get_constraint_details(),
            'variables': self._get_variable_details(),
            'lagrange_multipliers': self.get_shadow_prices(),
            'reduced_costs': self.get_reduced_costs(),
            'solver_stats': self.solver_stats
        }
    
    def _get_facility_details(self) -> Dict:
        return {
            'selected_facilities': self.selected_facilities,
            'total_modules': sum(self.selected_facilities.values()) if self.capacitated else len(self.selected_facilities),
            'locations': list(self.selected_facilities.keys()),
            'capacity_utilization': self._get_capacity_utilization() if self.capacitated else None
        }
    
    def _get_flow_details(self) -> Dict:
        total_flow = sum(self.flows.values())
        covered_flow = sum(coverage.get('covered_volume', 0) for coverage in self.flow_coverage.values())
        
        return {
            'total_flows': len(self.flows),
            'total_volume': total_flow,
            'covered_volume': covered_flow,
            'coverage_percentage': covered_flow / total_flow if total_flow > 0 else 0,
            'flow_breakdown': self.flow_coverage,
            'uncovered_flows': [od for od, cov in self.flow_coverage.items() 
                               if cov.get('covered_proportion', 0) < 0.01]
        }
    
    def _get_constraint_details(self) -> Dict:
        active_constraints = {name: info for name, info in self.constraints.items() 
                             if info.get('active', False)}
        
        return {
            'total_constraints': len(self.constraints),
            'active_constraints': len(active_constraints),
            'constraint_info': self.constraints,
            'binding_constraints': list(active_constraints.keys())
        }
    
    def _get_variable_details(self) -> Dict:
        facility_vars = {name: value for name, value in self.variables.items() 
                        if name.startswith(('x_', 'z_')) and value > 1e-6}
        flow_vars = {name: value for name, value in self.variables.items() 
                    if name.startswith('y_') and value > 1e-6}
        
        return {
            'total_variables': len(self.variables),
            'facility_variables': facility_vars,
            'flow_variables': flow_vars,
            'nonzero_variables': len([v for v in self.variables.values() if abs(v) > 1e-6])
        }
    
    def _get_capacity_utilization(self) -> Dict:
        if not self.capacitated:
            return None
        
        utilization = {}
        for site, modules in self.selected_facilities.items():
            total_capacity = modules * self.capacity
            used_capacity = self._calculate_facility_utilization(site)
            utilization[site] = {
                'modules': modules,
                'total_capacity': total_capacity,
                'used_capacity': used_capacity,
                'utilization_rate': used_capacity / total_capacity if total_capacity > 0 else 0
            }
        
        return utilization
    
    def print_problem_details(self):
        print("\n" + "="*80)
        print("OPTIMIZATION PROBLEM DETAILS")
        print("="*80)
        
        details = self.get_solution_details()
        
        print(f"Model Type: {details['model_type'].title()}")
        print(f"Solver: {details['solver_type'].upper()}")
        print(f"Status: {details['optimization_status']}")
        print(f"Objective Value: {details['objective_value']:.4f}")
        print(f"Solution Time: {details['solution_time']:.4f} seconds")

        print(f"\nProblem Size:")
        print(f"  Variables: {details['variables']['total_variables']}")
        print(f"  Constraints: {details['constraints']['total_constraints']}")
        print(f"  Candidate Sites: {len(self.candidate_sites)}")
        print(f"  OD Flows: {details['flows']['total_flows']}")
 
        print(f"\nSolution Quality:")
        print(f"  Flow Coverage: {details['flows']['coverage_percentage']:.2%}")
        print(f"  Facilities Used: {details['facilities']['total_modules']}")
        print(f"  Active Constraints: {details['constraints']['active_constraints']}")

        lagrange_mults = details['lagrange_multipliers']
        if lagrange_mults:
            significant_duals = {k: v for k, v in lagrange_mults.items() if abs(v) > 1e-6}
            print(f"  Significant Lagrange Multipliers: {len(significant_duals)}")
        
        print("="*80)
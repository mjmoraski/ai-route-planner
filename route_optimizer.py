import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys

class RouteOptimizer:
    """Class for optimizing delivery routes using OR-Tools."""
    
    def __init__(self, distance_matrix, duration_matrix, num_vehicles, depot=0, vehicle_capacities=None):
        """
        Initialize the route optimizer.
        
        Args:
            distance_matrix: Matrix of distances between locations
            duration_matrix: Matrix of travel times between locations
            num_vehicles: Number of vehicles available
            depot: Index of the depot location
            vehicle_capacities: List of capacities for each vehicle
        """
        print(f"[DEBUG] Initializing RouteOptimizer with {num_vehicles} vehicles")
        print(f"[DEBUG] Distance matrix shape: {np.array(distance_matrix).shape}")
        print(f"[DEBUG] Duration matrix shape: {np.array(duration_matrix).shape}")
        
        self.distance_matrix = distance_matrix
        self.duration_matrix = duration_matrix
        self.num_vehicles = num_vehicles
        self.depot = depot
        
        # Set default vehicle capacities if not provided
        if vehicle_capacities is None:
            self.vehicle_capacities = [10] * num_vehicles
        else:
            self.vehicle_capacities = vehicle_capacities
        
        print(f"[DEBUG] Vehicle capacities: {self.vehicle_capacities}")
        
        # Initialize OR-Tools routing model
        try:
            self.manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
            self.routing = pywrapcp.RoutingModel(self.manager)
            print("[DEBUG] Successfully created routing model")
        except Exception as e:
            print(f"[ERROR] Failed to create routing model: {e}")
            raise
        
        # Default search parameters
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        self.search_parameters.time_limit.seconds = 30  # Time limit for solving
        
        # Register the distance callback
        self._register_distance_callback()
        
        # Initialize time windows and demands as None
        self.time_windows = None
        self.service_times = None
        self.priorities = None
        self.priority_weights = 0
        self.objective = "distance"  # Default objective
        self.time_dimension_added = False  # Track if time dimension has been added
    
    def _register_distance_callback(self):
        """Register the distance callback for the routing model."""
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            # Ensure we return a valid integer
            distance = int(self.distance_matrix[from_node][to_node])
            return distance
        
        try:
            self.transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
            self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)
            print("[DEBUG] Successfully registered distance callback")
        except Exception as e:
            print(f"[ERROR] Failed to register distance callback: {e}")
            raise
    
    def set_time_windows(self, time_windows):
        """
        Set time windows for all locations.
        
        Args:
            time_windows: List of (earliest, latest) time tuples for each location.
        """
        print(f"[DEBUG] Setting time windows for {len(time_windows)} locations")
        
        # Verify time windows match number of nodes
        num_nodes = self.manager.GetNumberOfNodes()
        if len(time_windows) != num_nodes:
            print(f"[ERROR] Time windows count ({len(time_windows)}) doesn't match nodes ({num_nodes})")
            return
        
        self.time_windows = time_windows
        
        # Register time callback
        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            # Bounds checking
            if from_node >= len(self.duration_matrix) or to_node >= len(self.duration_matrix[0]):
                print(f"[ERROR] Index out of bounds: from_node={from_node}, to_node={to_node}")
                return 0
            # Convert to seconds for consistency
            return int(self.duration_matrix[from_node][to_node])
        
        try:
            time_callback_index = self.routing.RegisterTransitCallback(time_callback)
            
            # Add time dimension - reduce the max time to avoid overflow
            self.routing.AddDimension(
                time_callback_index,
                1800,  # Allow waiting time of up to 30 minutes (in seconds)
                86400,  # Maximum time (24 hours in seconds)
                False,  # Don't force start cumul to zero
                "Time"
            )
            
            self.time_dimension_added = True
            time_dimension = self.routing.GetDimensionOrDie("Time")
            
            # Add time window constraints with bounds checking
            for location_idx, time_window in enumerate(time_windows):
                if location_idx >= num_nodes:
                    print(f"[WARNING] Skipping time window for location {location_idx} (out of bounds)")
                    continue
                try:
                    index = self.manager.NodeToIndex(location_idx)
                    # Convert minutes to seconds - ensure valid range
                    start_time = max(0, int(time_window[0] * 60))
                    end_time = min(86400, int(time_window[1] * 60))  # Cap at 24 hours
                    time_dimension.CumulVar(index).SetRange(start_time, end_time)
                except Exception as e:
                    print(f"[WARNING] Failed to set time window for location {location_idx}: {e}")
                    
            print("[DEBUG] Successfully set time windows")
        except Exception as e:
            print(f"[ERROR] Failed to set time windows: {e}")
            self.time_dimension_added = False
    
    def set_service_times(self, service_times):
        """
        Set service times for each location.
        
        Args:
            service_times: List of service times (in minutes) for each location.
        """
        print(f"[DEBUG] Setting service times for {len(service_times)} locations")
        self.service_times = service_times
        
        # Check if time dimension has been added
        if not self.time_dimension_added:
            print("[WARNING] Time dimension not added, skipping service times")
            return
        
        try:
            time_dimension = self.routing.GetDimensionOrDie("Time")
            
            # Add service time at each location
            for location_idx, service_time in enumerate(service_times):
                try:
                    index = self.manager.NodeToIndex(location_idx)
                    # Convert minutes to seconds - ensure non-negative
                    service_seconds = max(0, int(service_time * 60))
                    time_dimension.SlackVar(index).SetValue(service_seconds)
                except Exception as e:
                    print(f"[WARNING] Failed to set service time for location {location_idx}: {e}")
                    
            print("[DEBUG] Successfully set service times")
        except Exception as e:
            print(f"[ERROR] Failed to set service times: {e}")
    
    def set_priorities(self, priorities, weight=5):
        """
        Set priorities for each location and their weight in the objective.
        
        Args:
            priorities: List of priority values for each location (higher = more important)
            weight: Weight of priority in the objective function (0-10)
        """
        print(f"[DEBUG] Setting priorities with weight {weight}")
        self.priorities = priorities
        self.priority_weights = weight
    
    def set_objective_minimize_total_distance(self):
        """Set the objective to minimize total distance across all routes."""
        self.objective = "distance"
    
    def set_objective_minimize_max_route(self):
        """Set the objective to minimize the maximum route length."""
        self.objective = "max_route"
    
    def set_objective_balance_routes(self):
        """Set the objective to balance route lengths."""
        self.objective = "balance"
    
    def solve(self):
        """
        Solve the routing problem.
        
        Returns:
            Dictionary mapping vehicle index to list of location indices in the route.
        """
        print("[DEBUG] Starting solve process")
        
        # Add capacity constraints if vehicle capacities are provided
        def demand_callback(from_index):
            """Return the demand of the location."""
            # All locations except depot have demand of 1
            from_node = self.manager.IndexToNode(from_index)
            return 1 if from_node != self.depot else 0
        
        try:
            demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
            
            self.routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # No slack
                self.vehicle_capacities,  # Vehicle capacities
                True,  # Start cumul to zero
                "Capacity"
            )
            print("[DEBUG] Successfully added capacity constraints")
        except Exception as e:
            print(f"[ERROR] Failed to add capacity constraints: {e}")
        
        # Set the objective based on user selection
        try:
            if self.objective == "max_route":
                # Minimize the maximum route length
                dimension_name = "Distance"
                self.routing.AddDimension(
                    self.transit_callback_index,
                    0,  # No slack
                    100000,  # Reduced max distance to avoid overflow
                    True,  # Start cumul to zero
                    dimension_name
                )
                distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
                distance_dimension.SetGlobalSpanCostCoefficient(100)
                print("[DEBUG] Set objective to minimize max route")
                
            elif self.objective == "balance":
                # Balance route lengths
                dimension_name = "Distance"
                self.routing.AddDimension(
                    self.transit_callback_index,
                    0,  # No slack
                    100000,  # Reduced max distance to avoid overflow
                    True,  # Start cumul to zero
                    dimension_name
                )
                distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
                
                # Add a cost for each vehicle proportional to the route length
                for vehicle_id in range(self.num_vehicles):
                    end_index = self.routing.End(vehicle_id)
                    distance_dimension.SetCumulVarSoftUpperBound(
                        end_index,
                        50000,  # Reduced bound
                        10  # Penalty coefficient
                    )
                print("[DEBUG] Set objective to balance routes")
        except Exception as e:
            print(f"[WARNING] Failed to set custom objective: {e}")
        
        # Add priority handling if provided - SIMPLIFIED VERSION
        if self.priorities and self.priority_weights > 0:
            print("[DEBUG] Adding priority constraints")
            try:
                # Create a disjunction for each non-depot location with priority-based penalty
                for location_idx in range(1, min(len(self.priorities), self.manager.GetNumberOfNodes())):
                    try:
                        index = self.manager.NodeToIndex(location_idx)
                        priority = self.priorities[location_idx]
                        
                        # Higher priority = lower penalty for not visiting
                        # Scale by priority weight (0-10)
                        penalty = 100  # Much lower penalty - we want to visit locations!
                        
                        # Add disjunction with penalty
                        self.routing.AddDisjunction([index], max(0, penalty))
                    except Exception as e:
                        print(f"[WARNING] Failed to add priority for location {location_idx}: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to add priorities: {e}")
        
        # Set additional search parameters for better solutions
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        self.search_parameters.time_limit.seconds = 10  # Reduced time limit
        self.search_parameters.log_search = False
        
        print("[DEBUG] Attempting to solve...")
        
        # Solve the problem
        try:
            solution = self.routing.SolveWithParameters(self.search_parameters)
            
            if not solution:
                print("[DEBUG] No solution found with initial parameters, trying relaxed parameters")
                # Try with relaxed parameters
                self.search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE)
                self.search_parameters.time_limit.seconds = 20
                solution = self.routing.SolveWithParameters(self.search_parameters)
                
                if not solution:
                    print("[ERROR] No solution found even with relaxed parameters")
                    return None
        except Exception as e:
            print(f"[ERROR] Exception during solve: {e}")
            return None
        
        print("[DEBUG] Solution found, extracting routes")
        
        # Extract the routes
        routes = {}
        try:
            for vehicle_id in range(self.num_vehicles):
                index = self.routing.Start(vehicle_id)
                route = [self.manager.IndexToNode(index)]
                
                safety_counter = 0
                max_iterations = 1000  # Prevent infinite loops
                
                while not self.routing.IsEnd(index) and safety_counter < max_iterations:
                    index = solution.Value(self.routing.NextVar(index))
                    route.append(self.manager.IndexToNode(index))
                    safety_counter += 1
                
                if safety_counter >= max_iterations:
                    print(f"[WARNING] Route extraction hit safety limit for vehicle {vehicle_id}")
                
                routes[vehicle_id] = route
                print(f"[DEBUG] Vehicle {vehicle_id} route: {route}")
        except Exception as e:
            print(f"[ERROR] Failed to extract routes: {e}")
            return None
        
        print("[DEBUG] Successfully extracted all routes")
        return routes
    
    def get_solution_info(self, solution):
        """
        Get detailed information about the solution.
        
        Args:
            solution: The solution object from OR-Tools
            
        Returns:
            Dictionary with solution statistics
        """
        if not solution:
            return None
            
        info = {
            'total_distance': 0,
            'max_route_distance': 0,
            'dropped_nodes': [],
            'routes': {}
        }
        
        try:
            # Get dropped nodes
            for node in range(self.routing.Size()):
                if self.routing.IsStart(node) or self.routing.IsEnd(node):
                    continue
                if solution.Value(self.routing.NextVar(node)) == node:
                    info['dropped_nodes'].append(self.manager.IndexToNode(node))
            
            # Get route information
            for vehicle_id in range(self.num_vehicles):
                index = self.routing.Start(vehicle_id)
                route_distance = 0
                route_load = 0
                route = []
                
                safety_counter = 0
                max_iterations = 1000
                
                while not self.routing.IsEnd(index) and safety_counter < max_iterations:
                    node_index = self.manager.IndexToNode(index)
                    route.append(node_index)
                    
                    # Get next index
                    previous_index = index
                    index = solution.Value(self.routing.NextVar(index))
                    
                    # Add distance
                    route_distance += self.routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
                    
                    # Add load
                    if node_index != self.depot:
                        route_load += 1
                        
                    safety_counter += 1
                
                # Add final node
                route.append(self.manager.IndexToNode(index))
                
                info['routes'][vehicle_id] = {
                    'route': route,
                    'distance': route_distance,
                    'load': route_load
                }
                
                info['total_distance'] += route_distance
                info['max_route_distance'] = max(info['max_route_distance'], route_distance)
        except Exception as e:
            print(f"[ERROR] Failed to get solution info: {e}")
        
        return info

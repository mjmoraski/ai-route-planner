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
        self.num_locations = len(distance_matrix)
        
        # Set default vehicle capacities if not provided
        if vehicle_capacities is None:
            self.vehicle_capacities = [10] * num_vehicles
        else:
            self.vehicle_capacities = vehicle_capacities
        
        print(f"[DEBUG] Vehicle capacities: {self.vehicle_capacities}")
        print(f"[DEBUG] Number of locations: {self.num_locations}")
        
        # Calculate total deliveries vs total capacity
        total_deliveries = self.num_locations - 1  # Exclude depot
        total_capacity = sum(self.vehicle_capacities)
        print(f"[DEBUG] Total deliveries: {total_deliveries}, Total capacity: {total_capacity}")
        
        if total_deliveries > total_capacity:
            print(f"[WARNING] Not enough capacity ({total_capacity}) for all deliveries ({total_deliveries})")
        
        # Initialize OR-Tools routing model
        try:
            self.manager = pywrapcp.RoutingIndexManager(
                self.num_locations, 
                num_vehicles, 
                depot
            )
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
        self.search_parameters.time_limit.seconds = 30
        
        # Register the distance callback
        self._register_distance_callback()
        
        # Initialize other variables
        self.time_windows = None
        self.service_times = None
        self.priorities = None
        self.priority_weights = 0
        self.objective = "distance"
        self.time_dimension_added = False
    
    def _register_distance_callback(self):
        """Register the distance callback for the routing model."""
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node])
        
        try:
            self.transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
            self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)
            print("[DEBUG] Successfully registered distance callback")
        except Exception as e:
            print(f"[ERROR] Failed to register distance callback: {e}")
            raise
    
    def set_time_windows(self, time_windows):
        """Set time windows for all locations."""
        print(f"[DEBUG] Setting time windows for {len(time_windows)} locations")
        
        if len(time_windows) != self.num_locations:
            print(f"[ERROR] Time windows count ({len(time_windows)}) doesn't match locations ({self.num_locations})")
            return
        
        self.time_windows = time_windows
        
        # Register time callback
        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(self.duration_matrix[from_node][to_node])
        
        try:
            time_callback_index = self.routing.RegisterTransitCallback(time_callback)
            
            # Add time dimension
            self.routing.AddDimension(
                time_callback_index,
                1800,  # Allow waiting time of up to 30 minutes
                86400,  # Maximum time (24 hours)
                False,  # Don't force start cumul to zero
                "Time"
            )
            
            self.time_dimension_added = True
            time_dimension = self.routing.GetDimensionOrDie("Time")
            
            # Add time window constraints
            for location_idx, time_window in enumerate(time_windows):
                try:
                    index = self.manager.NodeToIndex(location_idx)
                    start_time = max(0, int(time_window[0] * 60))
                    end_time = min(86400, int(time_window[1] * 60))
                    time_dimension.CumulVar(index).SetRange(start_time, end_time)
                except Exception as e:
                    print(f"[WARNING] Failed to set time window for location {location_idx}: {e}")
                    
            print("[DEBUG] Successfully set time windows")
        except Exception as e:
            print(f"[ERROR] Failed to set time windows: {e}")
            self.time_dimension_added = False
    
    def set_service_times(self, service_times):
        """Set service times for each location."""
        print(f"[DEBUG] Setting service times for {len(service_times)} locations")
        self.service_times = service_times
        
        if not self.time_dimension_added:
            print("[WARNING] Time dimension not added, skipping service times")
            return
        
        try:
            time_dimension = self.routing.GetDimensionOrDie("Time")
            
            for location_idx, service_time in enumerate(service_times):
                try:
                    index = self.manager.NodeToIndex(location_idx)
                    service_seconds = max(0, int(service_time * 60))
                    time_dimension.SlackVar(index).SetValue(service_seconds)
                except Exception as e:
                    print(f"[WARNING] Failed to set service time for location {location_idx}: {e}")
                    
            print("[DEBUG] Successfully set service times")
        except Exception as e:
            print(f"[ERROR] Failed to set service times: {e}")
    
    def set_priorities(self, priorities, weight=5):
        """Set priorities for each location."""
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
        """Solve the routing problem."""
        print("[DEBUG] Starting solve process")
        
        # Add capacity constraints
        def demand_callback(from_index):
            """Return the demand of the location."""
            from_node = self.manager.IndexToNode(from_index)
            return 0 if from_node == self.depot else 1
        
        try:
            demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
            
            self.routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # No slack
                self.vehicle_capacities,
                True,  # Start cumul to zero
                "Capacity"
            )
            print("[DEBUG] Successfully added capacity constraints")
        except Exception as e:
            print(f"[ERROR] Failed to add capacity constraints: {e}")
        
        # CRITICAL FIX: Instead of making locations optional, we'll make sure they're required
        # Only make locations optional if we don't have enough capacity
        total_deliveries = self.num_locations - 1
        total_capacity = sum(self.vehicle_capacities)
        
        if total_deliveries > total_capacity:
            print(f"[DEBUG] Making some locations optional due to capacity constraints")
            # Only make the lowest priority locations optional
            if self.priorities:
                # Sort locations by priority (excluding depot)
                location_priorities = [(i, self.priorities[i]) for i in range(1, self.num_locations)]
                location_priorities.sort(key=lambda x: x[1])  # Sort by priority (low to high)
                
                # Make only the excess locations optional
                excess = total_deliveries - total_capacity
                for i in range(excess):
                    location_idx = location_priorities[i][0]
                    index = self.manager.NodeToIndex(location_idx)
                    penalty = 10000  # High penalty - only skip if absolutely necessary
                    self.routing.AddDisjunction([index], penalty)
                    print(f"[DEBUG] Made location {location_idx} optional with penalty {penalty}")
            else:
                # If no priorities, make the last few locations optional
                excess = total_deliveries - total_capacity
                for location_idx in range(self.num_locations - excess, self.num_locations):
                    index = self.manager.NodeToIndex(location_idx)
                    penalty = 10000
                    self.routing.AddDisjunction([index], penalty)
                    print(f"[DEBUG] Made location {location_idx} optional with penalty {penalty}")
        else:
            print("[DEBUG] All locations should be visitable - not making any optional")
        
        # Set search parameters
        self.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        self.search_parameters.time_limit.seconds = 30
        self.search_parameters.log_search = False
        
        print("[DEBUG] Attempting to solve...")
        
        # Solve the problem
        try:
            solution = self.routing.SolveWithParameters(self.search_parameters)
            
            if not solution:
                print("[DEBUG] No solution found, trying with different strategy")
                # Try different strategies
                strategies = [
                    routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
                    routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
                    routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
                    routing_enums_pb2.FirstSolutionStrategy.SWEEP,
                ]
                
                for strategy in strategies:
                    print(f"[DEBUG] Trying strategy: {strategy}")
                    self.search_parameters.first_solution_strategy = strategy
                    self.search_parameters.time_limit.seconds = 60
                    solution = self.routing.SolveWithParameters(self.search_parameters)
                    if solution:
                        print(f"[DEBUG] Found solution with strategy: {strategy}")
                        break
                
                if not solution:
                    print("[ERROR] No solution found with any strategy")
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
                
                while not self.routing.IsEnd(index):
                    index = solution.Value(self.routing.NextVar(index))
                    route.append(self.manager.IndexToNode(index))
                
                routes[vehicle_id] = route
                stops = len(route) - 2  # Exclude start and end depot
                print(f"[DEBUG] Vehicle {vehicle_id} route: {route} ({stops} stops)")
        except Exception as e:
            print(f"[ERROR] Failed to extract routes: {e}")
            return None
        
        # Check if we actually visited any delivery locations
        total_stops = sum(len(route) - 2 for route in routes.values())
        print(f"[DEBUG] Total delivery stops across all routes: {total_stops}")
        
        if total_stops == 0:
            print("[WARNING] No delivery locations visited! This suggests a problem with the model setup.")
            
            # Try a very simple approach - force at least one delivery per vehicle that has capacity
            print("[DEBUG] Attempting simple fallback solution...")
            return self._create_simple_fallback_solution()
        
        print("[DEBUG] Successfully extracted all routes")
        return routes
    
    def _create_simple_fallback_solution(self):
        """Create a simple fallback solution that assigns deliveries round-robin."""
        print("[DEBUG] Creating simple fallback solution")
        
        routes = {}
        delivery_locations = list(range(1, self.num_locations))  # Exclude depot
        
        # Initialize empty routes
        for vehicle_id in range(self.num_vehicles):
            routes[vehicle_id] = [self.depot, self.depot]  # Start and end at depot
        
        # Assign deliveries round-robin to vehicles with capacity
        vehicle_loads = [0] * self.num_vehicles
        
        for i, location in enumerate(delivery_locations):
            vehicle_id = i % self.num_vehicles
            
            # Check if this vehicle has capacity
            if vehicle_loads[vehicle_id] < self.vehicle_capacities[vehicle_id]:
                # Insert the delivery location before the final depot
                routes[vehicle_id].insert(-1, location)
                vehicle_loads[vehicle_id] += 1
                print(f"[DEBUG] Assigned location {location} to vehicle {vehicle_id}")
        
        # Show the fallback solution
        for vehicle_id, route in routes.items():
            stops = len(route) - 2
            print(f"[DEBUG] Fallback - Vehicle {vehicle_id}: {route} ({stops} stops)")
        
        return routes
    
    def get_solution_info(self, solution):
        """Get detailed information about the solution."""
        if not solution:
            return None
            
        info = {
            'total_distance': 0,
            'max_route_distance': 0,
            'dropped_nodes': [],
            'routes': {}
        }
        
        try:
            # Calculate route distances
            for vehicle_id, route in solution.items():
                route_distance = 0
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    route_distance += self.distance_matrix[from_node][to_node]
                
                info['routes'][vehicle_id] = {
                    'route': route,
                    'distance': route_distance,
                    'stops': len(route) - 2
                }
                
                info['total_distance'] += route_distance
                info['max_route_distance'] = max(info['max_route_distance'], route_distance)
        except Exception as e:
            print(f"[ERROR] Failed to get solution info: {e}")
        
        return info

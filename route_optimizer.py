import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
        self.distance_matrix = distance_matrix
        self.duration_matrix = duration_matrix
        self.num_vehicles = num_vehicles
        self.depot = depot
        
        # Set default vehicle capacities if not provided
        if vehicle_capacities is None:
            self.vehicle_capacities = [10] * num_vehicles
        else:
            self.vehicle_capacities = vehicle_capacities
        
        # Initialize OR-Tools routing model
        self.manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
        self.routing = pywrapcp.RoutingModel(self.manager)
        
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
            return int(self.distance_matrix[from_node][to_node])
        
        self.transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)
    
    def set_time_windows(self, time_windows):
        """
        Set time windows for all locations.
        
        Args:
            time_windows: List of (earliest, latest) time tuples for each location.
        """
        self.time_windows = time_windows
        
        # Register time callback
        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            # Convert to seconds for consistency
            return int(self.duration_matrix[from_node][to_node])
        
        time_callback_index = self.routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension
        self.routing.AddDimension(
            time_callback_index,
            30 * 60,  # Allow waiting time of up to 30 minutes
            24 * 60 * 60,  # Maximum time (24 hours in seconds)
            False,  # Don't force start cumul to zero
            "Time"
        )
        
        self.time_dimension_added = True
        time_dimension = self.routing.GetDimensionOrDie("Time")
        
        # Add time window constraints
        for location_idx, time_window in enumerate(time_windows):
            index = self.manager.NodeToIndex(location_idx)
            # Convert minutes to seconds
            time_dimension.CumulVar(index).SetRange(
                int(time_window[0] * 60),  # Earliest in seconds
                int(time_window[1] * 60)   # Latest in seconds
            )
    
    def set_service_times(self, service_times):
        """
        Set service times for each location.
        
        Args:
            service_times: List of service times (in minutes) for each location.
        """
        self.service_times = service_times
        
        # Check if time dimension has been added
        if not self.time_dimension_added:
            raise ValueError("Must call set_time_windows before set_service_times")
        
        time_dimension = self.routing.GetDimensionOrDie("Time")
        
        # Add service time at each location
        for location_idx, service_time in enumerate(service_times):
            index = self.manager.NodeToIndex(location_idx)
            # Convert minutes to seconds
            time_dimension.SlackVar(index).SetValue(int(service_time * 60))
    
    def set_priorities(self, priorities, weight=5):
        """
        Set priorities for each location and their weight in the objective.
        
        Args:
            priorities: List of priority values for each location (higher = more important)
            weight: Weight of priority in the objective function (0-10)
        """
        self.priorities = priorities
        self.priority_weights = weight
        
        # We'll use this in the solve method to adjust the objective
    
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
        # Add capacity constraints if vehicle capacities are provided
        def demand_callback(from_index):
            """Return the demand of the location."""
            # All locations except depot have demand of 1
            from_node = self.manager.IndexToNode(from_index)
            return 1 if from_node != self.depot else 0
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # No slack
            self.vehicle_capacities,  # Vehicle capacities
            True,  # Start cumul to zero
            "Capacity"
        )
        
        # Set the objective based on user selection
        if self.objective == "max_route":
            # Minimize the maximum route length
            dimension_name = "Distance"
            self.routing.AddDimension(
                self.transit_callback_index,
                0,  # No slack
                3000000,  # Large max distance
                True,  # Start cumul to zero
                dimension_name
            )
            distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(100)
            
        elif self.objective == "balance":
            # Balance route lengths
            dimension_name = "Distance"
            self.routing.AddDimension(
                self.transit_callback_index,
                0,  # No slack
                3000000,  # Large max distance
                True,  # Start cumul to zero
                dimension_name
            )
            distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
            
            # Add a cost for each vehicle proportional to the route length
            for vehicle_id in range(self.num_vehicles):
                end_index = self.routing.End(vehicle_id)
                distance_dimension.SetCumulVarSoftUpperBound(
                    end_index,
                    3000000,
                    10  # Penalty coefficient
                )
        
        # Add priority handling if provided
        if self.priorities and self.priority_weights > 0:
            # Create a disjunction for each non-depot location with priority-based penalty
            for location_idx in range(1, len(self.priorities)):  # Skip depot
                index = self.manager.NodeToIndex(location_idx)
                priority = self.priorities[location_idx]
                
                # Higher priority = lower penalty for not visiting
                # Scale by priority weight (0-10)
                penalty = int(10000 * (4 - priority) * self.priority_weights / 10)
                
                # Add disjunction with penalty
                self.routing.AddDisjunction([index], penalty)
        
        # Set additional search parameters for better solutions
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        self.search_parameters.time_limit.seconds = 30
        self.search_parameters.log_search = False
        
        # Solve the problem
        solution = self.routing.SolveWithParameters(self.search_parameters)
        
        if not solution:
            # Try with relaxed parameters
            self.search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE)
            self.search_parameters.time_limit.seconds = 60
            solution = self.routing.SolveWithParameters(self.search_parameters)
            
            if not solution:
                return None
        
        # Extract the routes
        routes = {}
        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            route = [self.manager.IndexToNode(index)]
            
            while not self.routing.IsEnd(index):
                index = solution.Value(self.routing.NextVar(index))
                route.append(self.manager.IndexToNode(index))
            
            routes[vehicle_id] = route
        
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
            
            while not self.routing.IsEnd(index):
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
            
            # Add final node
            route.append(self.manager.IndexToNode(index))
            
            info['routes'][vehicle_id] = {
                'route': route,
                'distance': route_distance,
                'load': route_load
            }
            
            info['total_distance'] += route_distance
            info['max_route_distance'] = max(info['max_route_distance'], route_distance)
        
        return info

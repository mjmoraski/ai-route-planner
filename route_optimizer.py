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
        import numpy as np
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        
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
    
    def _register_distance_callback(self):
        """Register the distance callback for the routing model."""
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]
        
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
        
        time_dimension = self.routing.GetDimensionOrDie("Time")
        
        # Add time window constraints
        for location_idx, time_window in enumerate(time_windows):
            index = self.manager.NodeToIndex(location_idx)
            # Convert minutes to seconds
            time_dimension.CumulVar(index).SetRange(
                time_window[0] * 60,  # Earliest in seconds
                time_window[1] * 60   # Latest in seconds
            )
    
    def set_service_times(self, service_times):
        """
        Set service times for each location.
        
        Args:
            service_times: List of service times (in minutes) for each location.
        """
        self.service_times = service_times
        
        if not hasattr(self.routing, "GetDimensionOrDie"):
            raise ValueError("Must call set_time_windows before set_service_times")
        
        time_dimension = self.routing.GetDimensionOrDie("Time")
        
        # Add service time at each location
        for location_idx, service_time in enumerate(service_times):
            index = self.manager.NodeToIndex(location_idx)
            # Convert minutes to seconds
            time_dimension.SetTransitVar(index, int(service_time * 60))
    
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
            
            # Add a cost for each vehicle proportional to the square of the route length
            for vehicle_id in range(self.num_vehicles):
                end_index = self.routing.End(vehicle_id)
                self.routing.AddVariableMinimizedByFinalizer(
                    distance_dimension.CumulVar(end_index))
                
                # Add quadratic cost to balance routes
                coef = 10  # Weight of the balancing objective
                self.routing.AddToObjectiveFunction(
                    self.routing.VarPower(distance_dimension.CumulVar(end_index), 2) * coef)
        
        # Add priority handling if provided
        if self.priorities and self.priority_weights > 0:
            for location_idx, priority in enumerate(self.priorities):
                if location_idx == self.depot:
                    continue  # Skip depot
                    
                # Higher priority = lower cost (negative weight)
                for vehicle_id in range(self.num_vehicles):
                    index = self.manager.NodeToIndex(location_idx)
                    # Scale by priority weight (0-10)
                    priority_cost = -1000 * priority * self.priority_weights / 10
                    self.routing.AddToObjectiveFunction(
                        self.routing.ActiveVar(index) * priority_cost)
        
        # Solve the problem
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

def calculate_distance_matrix(self, locations):
    """
    Calculate distance and duration matrices using OSRM API.
    
    Args:
        locations: List of (lat, lon) tuples
        
    Returns:
        Tuple of (distance_matrix, duration_matrix)
    """
    import numpy as np
    import requests
    import time
    
    # Initialize matrices
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    duration_matrix = np.zeros((n, n))
    
    # Check if we have too many locations for a single API call
    # OSRM has a limit of around 100 locations per request
    if n > 100:
        print(f"Too many locations ({n}) for a single OSRM API call. Falling back to Euclidean distances.")
        distance_matrix = self.calculate_euclidean_distance_matrix(locations)
        # Estimate durations (assume 30 km/h average speed)
        duration_matrix = distance_matrix * 120  # seconds per km
        return distance_matrix, duration_matrix
    
    try:
        # OSRM API base URL (using the demo server)
        base_url = "https://router.project-osrm.org/table/v1/driving/"
        
        # Need to format locations as lon,lat (OSRM uses this order)
        locations_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
        
        # Add parameters
        params = "?annotations=distance,duration"
        
        # Build the complete URL
        url = base_url + locations_str + params
        
        print(f"Calling OSRM API with URL: {url}")
        
        # Make the API request with timeout
        response = requests.get(
            url, 
            headers={'User-Agent': 'AI-Route-Planner/1.0'},
            timeout=10  # Add a timeout to prevent hanging
        )
        
        # Check specifically for 404 and other errors
        if response.status_code == 404:
            print("OSRM API returned 404 Not Found. The API endpoint may be incorrect or the service may be down.")
            raise ValueError(f"OSRM API returned 404 Not Found: {response.text}")
        
        response.raise_for_status()
        
        data = response.json()
        
        # Extract matrices from response
        if 'distances' in data and 'durations' in data:
            distance_matrix = np.array(data['distances'])
            duration_matrix = np.array(data['durations'])
            print("Successfully retrieved distance and duration data from OSRM API")
        else:
            print(f"Invalid response format from OSRM API: {data}")
            raise ValueError("Invalid response from OSRM API - missing distances or durations")
    
    except requests.exceptions.Timeout:
        print("OSRM API request timed out. Falling back to Euclidean distances.")
        distance_matrix = self.calculate_euclidean_distance_matrix(locations)
        duration_matrix = distance_matrix * 120  # seconds per km
    
    except requests.exceptions.ConnectionError:
        print("Connection error when calling OSRM API. Falling back to Euclidean distances.")
        distance_matrix = self.calculate_euclidean_distance_matrix(locations)
        durati

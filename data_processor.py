def calculate_graphhopper_distance_matrix(self, locations, api_key):
    """
    Calculate distance and duration matrices using GraphHopper API.
    
    Args:
        locations: List of (lat, lon) tuples
        api_key: GraphHopper API key
        
    Returns:
        Tuple of (distance_matrix, duration_matrix)
    """
    import numpy as np
    import json
    import time
    
    # Initialize matrices
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    duration_matrix = np.zeros((n, n))
    
    if not api_key:
        print("No GraphHopper API key provided, falling back to Euclidean distances")
        return self.calculate_euclidean_distance_matrix(locations), self.calculate_euclidean_distance_matrix(locations) * 0.12
    
    # GraphHopper has a limit on the number of points in a single request
    # Free tier typically allows 10x10 matrix (100 elements) per request
    MAX_LOCATIONS_PER_REQUEST = 10
    
    # Process the distance matrix in batches
    for i in range(0, n, MAX_LOCATIONS_PER_REQUEST):
        i_end = min(i + MAX_LOCATIONS_PER_REQUEST, n)
        
        for j in range(0, n, MAX_LOCATIONS_PER_REQUEST):
            j_end = min(j + MAX_LOCATIONS_PER_REQUEST, n)
            
            # Skip if i == j (diagonal blocks)
            if i == j:
                # For diagonal elements (same location), set distance and time to 0
                for k in range(i, i_end):
                    distance_matrix[k, k] = 0
                    duration_matrix[k, k] = 0
                continue
            
            # Prepare points for this batch
            from_points = locations[i:i_end]
            to_points = locations[j:j_end]
            
            # Format points for GraphHopper API
            from_points_json = [{"lat": lat, "lng": lon} for lat, lon in from_points]
            to_points_json = [{"lat": lat, "lng": lon} for lat, lon in to_points]
            
            # Prepare the request
            url = "https://graphhopper.com/api/1/matrix"
            params = {
                "key": api_key,
                "out_arrays": ["distances", "times"],
                "vehicle": "car"
            }
            
            payload = {
                "from_points": from_points_json,
                "to_points": to_points_json,
                "fail_fast": False
            }
            
            try:
                # Make the API request
                response = self.requests.post(url, params=params, json=payload, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    print("Rate limited by GraphHopper API, waiting before retry...")
                    time.sleep(2)  # Wait before retrying
                    response = self.requests.post(url, params=params, json=payload, timeout=30)
                
                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract matrices
                    if 'distances' in data and 'times' in data:
                        batch_distances = np.array(data['distances'])
                        batch_times = np.array(data['times'])
                        
                        # Fill the corresponding part of the full matrices
                        for k_from, k_global_from in enumerate(range(i, i_end)):
                            for k_to, k_global_to in enumerate(range(j, j_end)):
                                distance_matrix[k_global_from, k_global_to] = batch_distances[k_from][k_to]
                                duration_matrix[k_global_from, k_global_to] = batch_times[k_from][k_to]
                    else:
                        print(f"Unexpected GraphHopper API response format: {data}")
                        # Fill with Euclidean distances for this batch
                        self._fill_block_with_euclidean(distance_matrix, duration_matrix, locations, range(i, i_end), range(j, j_end))
                else:
                    print(f"GraphHopper API error: {response.status_code} - {response.text}")
                    # Fill with Euclidean distances for this batch
                    self._fill_block_with_euclidean(distance_matrix, duration_matrix, locations, range(i, i_end), range(j, j_end))
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error calculating distances with GraphHopper: {e}")
                # Fill with Euclidean distances for this batch
                self._fill_block_with_euclidean(distance_matrix, duration_matrix, locations, range(i, i_end), range(j, j_end))
    
    return distance_matrix, duration_matrix

def _fill_block_with_euclidean(self, distance_matrix, duration_matrix, locations, from_indices, to_indices):
    """Helper method to fill a block with Euclidean distances when API fails."""
    for i in from_indices:
        for j in to_indices:
            if i != j:  # Skip diagonals
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                
                # Approximate distance calculation
                lat_km = abs(lat1 - lat2) * 111
                lon_km = abs(lon1 - lon2) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
                
                # Euclidean distance
                distance = np.sqrt(lat_km**2 + lon_km**2) * 1000  # Convert to meters
                
                distance_matrix[i, j] = distance
                duration_matrix[i, j] = distance * 0.12  # seconds per meter (30 km/h)

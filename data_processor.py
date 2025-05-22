import pandas as pd
import numpy as np
import requests
import time
import json

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_call = 0
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last_call < min_interval:
            time.sleep(min_interval - time_since_last_call)
        
        self.last_call = time.time()

class DataProcessor:
    """Class for processing delivery data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.requests = requests
        self.rate_limiter = RateLimiter(calls_per_second=1)
    
    def ensure_required_columns(self, df):
        """
        Ensure that the DataFrame has all required columns.
        
        Args:
            df: pandas DataFrame with delivery data
            
        Returns:
            DataFrame with all required columns (filled with defaults if missing)
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Add ID column if missing
        if 'ID' not in df_copy.columns:
            df_copy['ID'] = range(1, len(df_copy) + 1)
        
        # Add Name column if missing
        if 'Name' not in df_copy.columns:
            df_copy['Name'] = [f"Customer {i}" for i in range(1, len(df_copy) + 1)]
        
        # Add Time Window columns if missing
        if 'Time Window Start' not in df_copy.columns:
            df_copy['Time Window Start'] = '08:00'
        
        if 'Time Window End' not in df_copy.columns:
            df_copy['Time Window End'] = '18:00'
        
        # Add Service Time column if missing
        if 'Service Time (min)' not in df_copy.columns:
            df_copy['Service Time (min)'] = 15
        
        # Add Priority column if missing
        if 'Priority' not in df_copy.columns:
            df_copy['Priority'] = 'Medium'
        
        # Add Package Size column if missing
        if 'Package Size' not in df_copy.columns:
            df_copy['Package Size'] = 'Medium'
        
        return df_copy
    
    def geocode_addresses(self, df):
        """
        Geocode addresses to get latitude and longitude.
        
        Args:
            df: pandas DataFrame with Address column
            
        Returns:
            DataFrame with added Latitude and Longitude columns
        """
        # Check if Address column exists
        if 'Address' not in df.columns:
            return None
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Initialize latitude and longitude columns
        df_copy['Latitude'] = None
        df_copy['Longitude'] = None
        
        # Use Nominatim API for geocoding
        base_url = "https://nominatim.openstreetmap.org/search"
        
        for idx, row in df_copy.iterrows():
            if pd.isna(row['Address']) or not row['Address']:
                continue
                
            params = {
                'q': row['Address'],
                'format': 'json',
                'limit': 1
            }
            
            try:
                # Rate limit the requests
                self.rate_limiter.wait_if_needed()
                
                response = self.requests.get(base_url, params=params, headers={'User-Agent': 'AI-Route-Planner/1.0'})
                response.raise_for_status()
                
                data = response.json()
                
                if data:
                    df_copy.at[idx, 'Latitude'] = float(data[0]['lat'])
                    df_copy.at[idx, 'Longitude'] = float(data[0]['lon'])
            except Exception as e:
                print(f"Error geocoding address {row['Address']}: {e}")
        
        # Check if we got any results
        if df_copy['Latitude'].notna().sum() == 0:
            return None
            
        # Fill missing values with estimates
        if df_copy['Latitude'].isna().any():
            # Get the average of the available coordinates
            avg_lat = df_copy['Latitude'].dropna().mean()
            avg_lon = df_copy['Longitude'].dropna().mean()
            
            # Fill missing values with slightly perturbed averages
            for idx, row in df_copy[df_copy['Latitude'].isna()].iterrows():
                df_copy.at[idx, 'Latitude'] = avg_lat + np.random.normal(0, 0.01)
                df_copy.at[idx, 'Longitude'] = avg_lon + np.random.normal(0, 0.01)
        
        return df_copy
    
    def calculate_distance_matrix(self, locations):
        """
        Calculate distance and duration matrices using OSRM API.
        
        Args:
            locations: List of (lat, lon) tuples
            
        Returns:
            Tuple of (distance_matrix, duration_matrix)
        """
        # Initialize matrices
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        duration_matrix = np.zeros((n, n))
        
        # OSRM API base URL (using the demo server)
        base_url = "https://router.project-osrm.org/table/v1/driving/"
        
        # Need to format locations as lon,lat (OSRM uses this order)
        locations_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
        
        # Add parameters
        params = "?annotations=distance,duration"
        
        # Build the complete URL
        url = base_url + locations_str + params
        
        try:
            # Make the API request
            response = requests.get(url, headers={'User-Agent': 'AI-Route-Planner/1.0'})
            response.raise_for_status()
            
            data = response.json()
            
            # Extract matrices from response
            if 'distances' in data and 'durations' in data:
                distance_matrix = np.array(data['distances'])
                duration_matrix = np.array(data['durations'])
            else:
                raise ValueError("Invalid response from OSRM API")
        except Exception as e:
            print(f"Error calculating distances with OSRM: {e}")
            # Fall back to Euclidean distances
            distance_matrix = self.calculate_euclidean_distance_matrix(locations)
            # Estimate durations (assume 30 km/h average speed)
            duration_matrix = distance_matrix * 120  # seconds per km
        
        return distance_matrix, duration_matrix
    
    def calculate_euclidean_distance_matrix(self, locations):
        """
        Calculate Euclidean distance matrix as a fallback.
        
        Args:
            locations: List of (lat, lon) tuples
            
        Returns:
            Distance matrix
        """
        # Initialize the matrix
        n = len(locations)
        matrix = np.zeros((n, n))
        
        # Approximation: 1 degree of latitude = 111 km
        # Longitude conversion varies with latitude
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                
                # Approximate distance calculation
                lat_km = abs(lat1 - lat2) * 111
                lon_km = abs(lon1 - lon2) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
                
                # Euclidean distance
                distance = np.sqrt(lat_km**2 + lon_km**2) * 1000  # Convert to meters
                
                matrix[i, j] = distance
        
        return matrix
    
    def calculate_graphhopper_distance_matrix(self, locations, api_key):
        """
        Calculate distance and duration matrices using GraphHopper API.
        
        Args:
            locations: List of (lat, lon) tuples
            api_key: GraphHopper API key
            
        Returns:
            Tuple of (distance_matrix, duration_matrix)
        """
        # Initialize matrices
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        duration_matrix = np.zeros((n, n))
        
        if not api_key:
            print("No GraphHopper API key provided, falling back to Euclidean distances")
            return self.calculate_euclidean_distance_matrix(locations), self.calculate_euclidean_distance_matrix(locations) * 0.12
        
        # GraphHopper Matrix API has limits
        MAX_MATRIX_SIZE = 100  # Free tier limit
        total_elements = n * n
        
        if total_elements > MAX_MATRIX_SIZE:
            # For large matrices, use a hybrid approach
            # Calculate real distances for nearby locations, use estimates for distant ones
            print(f"Matrix too large ({total_elements} elements). Using hybrid approach.")
            
            # First, get Euclidean distances as a baseline
            euclidean_matrix = self.calculate_euclidean_distance_matrix(locations)
            
            # Sort pairs by Euclidean distance and get real distances for closest ones
            pairs = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pairs.append((i, j, euclidean_matrix[i][j]))
            
            # Sort by distance
            pairs.sort(key=lambda x: x[2])
            
            # Get real distances for the closest pairs up to the limit
            pairs_to_calculate = pairs[:MAX_MATRIX_SIZE - n]  # Reserve n for diagonal
            
            # Batch process the closest pairs
            batch_size = 10
            for batch_start in range(0, len(pairs_to_calculate), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs_to_calculate))
                batch_pairs = pairs_to_calculate[batch_start:batch_end]
                
                # Create unique points for this batch
                unique_indices = set()
                for i, j, _ in batch_pairs:
                    unique_indices.add(i)
                    unique_indices.add(j)
                
                unique_indices = sorted(list(unique_indices))
                batch_locations = [locations[i] for i in unique_indices]
                
                # Create mapping
                index_map = {idx: i for i, idx in enumerate(unique_indices)}
                
                # Calculate for this batch
                url = "https://graphhopper.com/api/1/matrix"
                params = {
                    "key": api_key,
                    "out_arrays": ["distances", "times"],
                    "vehicle": "car"
                }
                
                points_json = [{"lat": lat, "lng": lon} for lat, lon in batch_locations]
                payload = {
                    "points": points_json,
                    "fail_fast": False
                }
                
                try:
                    self.rate_limiter.wait_if_needed()
                    response = self.requests.post(url, params=params, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'distances' in data and 'times' in data:
                            batch_distances = np.array(data['distances'])
                            batch_times = np.array(data['times'])
                            
                            # Map back to original indices
                            for i_orig, j_orig, _ in batch_pairs:
                                i_batch = index_map[i_orig]
                                j_batch = index_map[j_orig]
                                
                                distance_matrix[i_orig, j_orig] = batch_distances[i_batch][j_batch]
                                duration_matrix[i_orig, j_orig] = batch_times[i_batch][j_batch]
                
                except Exception as e:
                    print(f"Error in batch calculation: {e}")
            
            # For remaining pairs, use scaled Euclidean distances
            for i in range(n):
                for j in range(n):
                    if distance_matrix[i, j] == 0 and i != j:
                        # Use a scaling factor based on calculated distances
                        scale_factor = 1.5  # Roads are typically 1.5x straight line
                        distance_matrix[i, j] = euclidean_matrix[i, j] * scale_factor
                        duration_matrix[i, j] = distance_matrix[i, j] * 0.06  # 60 km/h average
        
        else:
            # Small matrix - calculate all at once
            url = "https://graphhopper.com/api/1/matrix"
            params = {
                "key": api_key,
                "out_arrays": ["distances", "times"],
                "vehicle": "car"
            }
            
            points_json = [{"lat": lat, "lng": lon} for lat, lon in locations]
            payload = {
                "points": points_json,
                "fail_fast": False
            }
            
            try:
                self.rate_limiter.wait_if_needed()
                response = self.requests.post(url, params=params, json=payload, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    print("Rate limited by GraphHopper API, waiting before retry...")
                    time.sleep(2)  # Wait before retrying
                    response = self.requests.post(url, params=params, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'distances' in data and 'times' in data:
                        distance_matrix = np.array(data['distances'])
                        duration_matrix = np.array(data['times'])
                    else:
                        raise ValueError("Invalid response format")
                else:
                    raise ValueError(f"API error: {response.status_code}")
                    
            except Exception as e:
                print(f"Error calculating distances with GraphHopper: {e}")
                # Fall back to Euclidean distances
                distance_matrix = self.calculate_euclidean_distance_matrix(locations)
                duration_matrix = distance_matrix * 0.12
        
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

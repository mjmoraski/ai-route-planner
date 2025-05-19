class DataProcessor:
    """Class for processing delivery data."""
    
    def __init__(self):
        """Initialize the data processor."""
        import pandas as pd
        import numpy as np
        import requests
        
        self.requests = requests
    
    def ensure_required_columns(self, df):
        """
        Ensure that the DataFrame has all required columns.
        
        Args:
            df: pandas DataFrame with delivery data
            
        Returns:
            DataFrame with all required columns (filled with defaults if missing)
        """
        import pandas as pd
        
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
        import pandas as pd
        import time
        
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
                # Add a delay to respect API usage limits
                time.sleep(1)
                
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
            import numpy as np
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
        import numpy as np
        import requests
        import time
        
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
        import numpy as np
        
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

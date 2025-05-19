class MapVisualizer:
    """Class for visualizing routes on a map."""
    
    def __init__(self):
        """Initialize the map visualizer."""
        import folium
        import numpy as np
        
        self.folium = folium
        self.np = np
    
    def create_route_map(self, routes, locations, df=None):
        """
        Create a folium map with the optimized routes.
        
        Args:
            routes: Dictionary mapping vehicle index to a list of location indices
            locations: List of (lat, lon) tuples
            df: Optional DataFrame with additional information
            
        Returns:
            Folium map object
        """
        # Calculate the center of the map
        center_lat = sum(loc[0] for loc in locations) / len(locations)
        center_lon = sum(loc[1] for loc in locations) / len(locations)
        
        # Create a map
        route_map = self.folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Different colors for different vehicles
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Add routes
        for vehicle_id, route in routes.items():
            # Skip empty routes
            if len(route) <= 2:  # Just depot-depot
                continue
                
            # Get color for this vehicle
            color = colors[vehicle_id % len(colors)]
            
            # Create a feature group for this route
            route_group = self.folium.FeatureGroup(name=f"Vehicle {vehicle_id + 1}")
            
            # Add markers for each stop
            for i, location_idx in enumerate(route):
                lat, lon = locations[location_idx]
                
                # Determine marker type and popup content
                if i == 0 or i == len(route) - 1:
                    # Depot marker
                    icon = self.folium.Icon(color='black', icon='home', prefix='fa')
                    popup_content = f"Depot"
                else:
                    # Regular stop marker
                    icon = self.folium.Icon(color=color, icon='shopping-bag', prefix='fa')
                    
                    # Add info from DataFrame if available
                    if df is not None and location_idx > 0:
                        row = df.iloc[location_idx - 1]  # Adjust for depot offset
                        name = row.get('Name', f'Stop {i}')
                        
                        popup_content = f"<b>Stop {i}: {name}</b><br>"
                        
                        # Add optional fields if available
                        if 'ID' in row:
                            popup_content += f"ID: {row['ID']}<br>"
                        if 'Address' in row:
                            popup_content += f"Address: {row['Address']}<br>"
                        if 'Priority' in row:
                            popup_content += f"Priority: {row['Priority']}<br>"
                        if 'Time Window Start' in row and 'Time Window End' in row:
                            popup_content += f"Time Window: {row['Time Window Start']} - {row['Time Window End']}<br>"
                        if 'Service Time (min)' in row:
                            popup_content += f"Service Time: {row['Service Time (min)']} minutes<br>"
                        if 'Package Size' in row:
                            popup_content += f"Package Size: {row['Package Size']}<br>"
                    else:
                        popup_content = f"Stop {i}"
                
                # Add marker
                marker = self.folium.Marker(
                    location=[lat, lon],
                    popup=self.folium.Popup(popup_content, max_width=300),
                    icon=icon
                )
                route_group.add_child(marker)
            
            # Add polyline for the route
            route_points = [locations[idx] for idx in route]
            route_line = self.folium.PolyLine(
                locations=route_points,
                weight=4,
                color=color,
                opacity=0.8,
                tooltip=f"Vehicle {vehicle_id + 1}"
            )
            route_group.add_child(route_line)
            
            # Add the feature group to the map
            route_map.add_child(route_group)
        
        # Add Layer Control
        self.folium.LayerControl().add_to(route_map)
        
        # Add fullscreen control
        plugins = self.folium.plugins
        plugins.Fullscreen().add_to(route_map)
        
        # Add distance scale
        plugins.MeasureControl().add_to(route_map)
        
        return route_map

import folium
import numpy as np
import matplotlib.colors as mcolors
from folium import plugins

class MapVisualizer:
    """Class for visualizing routes on a map."""
    
    def __init__(self):
        """Initialize the map visualizer."""
        pass
    
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
        
        # Create a map with a nice default style
        route_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=12,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Different colors for different vehicles
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Add routes
        for vehicle_id, route in routes.items():
            # Skip empty routes
            if len(route) <= 2:  # Just depot-depot
                continue
                
            # Get color for this vehicle
            color = colors[vehicle_id % len(colors)]
            
            # Create a feature group for this route
            route_group = folium.FeatureGroup(name=f"Vehicle {vehicle_id + 1}")
            
            # Add markers for each stop
            for i, location_idx in enumerate(route):
                lat, lon = locations[location_idx]
                
                # Determine marker type and popup content
                if i == 0 or i == len(route) - 1:
                    # Depot marker
                    icon = folium.Icon(
                        color='black', 
                        icon='home', 
                        prefix='fa'
                    )
                    popup_content = "<b>🏢 Depot</b>"
                    tooltip_text = "Depot"
                else:
                    # Regular stop marker
                    icon = folium.Icon(
                        color=color.replace('#', '').lower() if color.startswith('#') else color, 
                        icon='shopping-bag', 
                        prefix='fa'
                    )
                    
                    # Add info from DataFrame if available
                    if df is not None and location_idx > 0:
                        row = df.iloc[location_idx - 1]  # Adjust for depot offset
                        name = row.get('Name', f'Stop {i}')
                        
                        popup_content = f"<b>📍 Stop {i}: {name}</b><br>"
                        
                        # Add optional fields if available
                        if 'ID' in row:
                            popup_content += f"<b>ID:</b> {row['ID']}<br>"
                        if 'Address' in row:
                            popup_content += f"<b>Address:</b> {row['Address']}<br>"
                        if 'Priority' in row:
                            priority_color = {
                                'High': '🔴',
                                'Medium': '🟡',
                                'Low': '🟢'
                            }.get(row['Priority'], '⚪')
                            popup_content += f"<b>Priority:</b> {priority_color} {row['Priority']}<br>"
                        if 'Time Window Start' in row and 'Time Window End' in row:
                            popup_content += f"<b>Time Window:</b> ⏰ {row['Time Window Start']} - {row['Time Window End']}<br>"
                        if 'Service Time (min)' in row:
                            popup_content += f"<b>Service Time:</b> ⏱️ {row['Service Time (min)']} minutes<br>"
                        if 'Package Size' in row:
                            package_icon = {
                                'Small': '📦',
                                'Medium': '📦📦',
                                'Large': '📦📦📦'
                            }.get(row['Package Size'], '📦')
                            popup_content += f"<b>Package Size:</b> {package_icon} {row['Package Size']}<br>"
                        
                        tooltip_text = f"Stop {i}: {name}"
                    else:
                        popup_content = f"<b>📍 Stop {i}</b>"
                        tooltip_text = f"Stop {i}"
                
                # Add marker with number
                if i > 0 and i < len(route) - 1:
                    # Add numbered marker for stops
                    marker = folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=tooltip_text,
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="
                                background-color: {color};
                                border: 2px solid white;
                                border-radius: 50%;
                                width: 30px;
                                height: 30px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-weight: bold;
                                color: white;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                            ">{i}</div>
                            """,
                            icon_size=(30, 30),
                            icon_anchor=(15, 15)
                        )
                    )
                else:
                    # Regular icon for depot
                    marker = folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=tooltip_text,
                        icon=icon
                    )
                route_group.add_child(marker)
            
            # Add polyline for the route with arrows
            route_points = [locations[idx] for idx in route]
            
            # Main route line
            route_line = folium.PolyLine(
                locations=route_points,
                weight=4,
                color=color,
                opacity=0.8,
                tooltip=f"Vehicle {vehicle_id + 1} Route"
            )
            route_group.add_child(route_line)
            
            # Add direction arrows
            for i in range(len(route_points) - 1):
                # Calculate midpoint
                mid_lat = (route_points[i][0] + route_points[i+1][0]) / 2
                mid_lon = (route_points[i][1] + route_points[i+1][1]) / 2
                
                # Add arrow marker
                folium.RegularPolygonMarker(
                    location=[mid_lat, mid_lon],
                    number_of_sides=3,
                    radius=8,
                    rotation=0,  # This would need to be calculated based on direction
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(route_group)
            
            # Add the feature group to the map
            route_map.add_child(route_group)
        
        # Add Layer Control
        folium.LayerControl(collapsed=False).add_to(route_map)
        
        # Add fullscreen control
        plugins.Fullscreen(
            position='topright',
            title='Expand',
            title_cancel='Exit',
            force_separate_button=True
        ).add_to(route_map)
        
        # Add distance measurement tool
        plugins.MeasureControl(
            position='bottomleft',
            primary_length_unit='kilometers',
            secondary_length_unit='miles',
            primary_area_unit='sqkilometers',
            secondary_area_unit='acres'
        ).add_to(route_map)
        
        # Add minimap
        minimap = plugins.MiniMap(
            toggle_display=True,
            tile_layer='OpenStreetMap',
            position='bottomright'
        )
        route_map.add_child(minimap)
        
        # Add a legend
        legend_html = self._create_legend(routes, colors)
        route_map.get_root().html.add_child(folium.Element(legend_html))
        
        return route_map
    
    def _create_legend(self, routes, colors):
        """Create an HTML legend for the map."""
        active_routes = [(vid, route) for vid, route in routes.items() if len(route) > 2]
        
        legend_items = []
        for vehicle_id, route in active_routes:
            color = colors[vehicle_id % len(colors)]
            stops = len(route) - 2  # Exclude depot visits
            legend_items.append(
                f'<li><span style="background:{color};width:15px;height:15px;'
                f'display:inline-block;margin-right:5px;border-radius:50%;"></span>'
                f'Vehicle {vehicle_id + 1} ({stops} stops)</li>'
            )
        
        legend_html = f'''
        <div style="
            position: fixed;
            top: 10px;
            right: 50px;
            background: white;
            padding: 10px;
            border: 2px solid grey;
            border-radius: 5px;
            z-index: 9999;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        ">
        <h4 style="margin: 0 0 5px 0;">Vehicle Routes</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
            {"".join(legend_items)}
        </ul>
        <hr style="margin: 10px 0 5px 0;">
        <div style="font-size: 12px;">
            <span style="margin-right: 10px;">🏢 Depot</span>
            <span>📍 Delivery Stop</span>
        </div>
        </div>
        '''
        
        return legend_html
    
    def create_simple_map(self, locations, df=None):
        """
        Create a simple map showing all locations without routes.
        
        Args:
            locations: List of (lat, lon) tuples
            df: Optional DataFrame with location information
            
        Returns:
            Folium map object
        """
        # Calculate the center of the map
        center_lat = sum(loc[0] for loc in locations) / len(locations)
        center_lon = sum(loc[1] for loc in locations) / len(locations)
        
        # Create a map
        simple_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each location
        for i, (lat, lon) in enumerate(locations):
            if i == 0:
                # Depot marker
                icon = folium.Icon(color='black', icon='home', prefix='fa')
                popup_content = "<b>Depot</b>"
            else:
                # Regular location marker
                icon = folium.Icon(color='blue', icon='map-marker', prefix='fa')
                
                if df is not None and i <= len(df):
                    row = df.iloc[i - 1]
                    name = row.get('Name', f'Location {i}')
                    popup_content = f"<b>{name}</b><br>"
                    if 'Address' in row:
                        popup_content += f"Address: {row['Address']}<br>"
                else:
                    popup_content = f"<b>Location {i}</b>"
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                icon=icon
            ).add_to(simple_map)
        
        return simple_map

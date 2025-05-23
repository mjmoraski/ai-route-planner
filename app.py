import streamlit as st
import pandas as pd
import numpy as np
# import openai  # Commented out OpenAI
import folium
from streamlit_folium import folium_static
import io
import time
import requests
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from PIL import Image
from fpdf import FPDF

# Import custom modules
from route_optimizer import RouteOptimizer
from data_processor import DataProcessor
from map_visualizer import MapVisualizer

# Set page config
st.set_page_config(page_title="Final Mile AI Planner", page_icon="🚚", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    div.stButton > button {width: 100%; height: 3em; font-weight: bold;}
    div.stDownloadButton > button {width: 100%; height: 3em;}
    .st-emotion-cache-16txtl3 h1 {margin-bottom: 0;}
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'optimized_routes' not in st.session_state:
    st.session_state.optimized_routes = None
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ai_explanation' not in st.session_state:
    st.session_state.ai_explanation = None
if 'map_html' not in st.session_state:
    st.session_state.map_html = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = {}

# Helper functions
def validate_delivery_data(df):
    """Validate the delivery data and return any issues found."""
    issues = []
    
    # Check for duplicate IDs
    if 'ID' in df.columns and df['ID'].duplicated().any():
        issues.append("Duplicate IDs found in the data")
    
    # Check for valid coordinates
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        invalid_coords = df[
            (df['Latitude'] < -90) | (df['Latitude'] > 90) |
            (df['Longitude'] < -180) | (df['Longitude'] > 180) |
            df['Latitude'].isna() | df['Longitude'].isna()
        ]
        if not invalid_coords.empty:
            issues.append(f"{len(invalid_coords)} rows have invalid or missing coordinates")
    
    # Check time windows
    if 'Time Window Start' in df.columns and 'Time Window End' in df.columns:
        try:
            for idx, row in df.iterrows():
                if pd.notna(row['Time Window Start']) and pd.notna(row['Time Window End']):
                    start_parts = str(row['Time Window Start']).split(':')
                    end_parts = str(row['Time Window End']).split(':')
                    
                    if len(start_parts) != 2 or len(end_parts) != 2:
                        issues.append(f"Invalid time format in row {idx}")
                        continue
                    
                    start_hour = int(start_parts[0])
                    start_min = int(start_parts[1])
                    end_hour = int(end_parts[0])
                    end_min = int(end_parts[1])
                    
                    # Check if end time is after start time
                    start_total = start_hour * 60 + start_min
                    end_total = end_hour * 60 + end_min
                    
                    if end_total <= start_total:
                        issues.append(f"Time window end before start in row {idx}")
        except Exception as e:
            issues.append(f"Error validating time windows: {e}")
    
    return issues

def generate_csv_template():
    """Generate a CSV template for users to fill in."""
    template_data = {
        'ID': [1, 2, 3],
        'Name': ['Customer A', 'Customer B', 'Customer C'],
        'Address': ['123 Main St, City, State', '456 Oak Ave, City, State', '789 Pine Rd, City, State'],
        'Latitude': [40.7128, 40.7260, 40.7360],
        'Longitude': [-74.0060, -73.9970, -73.9850],
        'Time Window Start': ['09:00', '10:00', '14:00'],
        'Time Window End': ['11:00', '12:00', '16:00'],
        'Service Time (min)': [15, 20, 15],
        'Priority': ['High', 'Medium', 'Low'],
        'Package Size': ['Small', 'Large', 'Medium']
    }
    
    return pd.DataFrame(template_data)

# Set up the page header and information
st.title("🚚 Final Mile AI Planner")
st.caption("🧠 Route optimization using OR-Tools (AI and GraphHopper features temporarily disabled)")

# Create sidebar for settings and configurations
with st.sidebar:
    st.header("Settings")
    
    st.info("📌 AI explanations are temporarily disabled")
    st.info("📌 Using Euclidean (straight-line) distances")
    
    st.subheader("Sample Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Sample Data"):
            # Create sample delivery data
            sample_data = {
                'ID': list(range(1, 11)),
                'Name': [f"Customer {i}" for i in range(1, 11)],
                'Address': [
                    "123 Main St, New York, NY",
                    "456 Elm St, New York, NY",
                    "789 Oak St, New York, NY",
                    "101 Pine St, New York, NY",
                    "202 Maple St, New York, NY",
                    "303 Cedar St, New York, NY",
                    "404 Birch St, New York, NY",
                    "505 Walnut St, New York, NY",
                    "606 Cherry St, New York, NY",
                    "707 Spruce St, New York, NY"
                ],
                'Latitude': [40.7128 + i*0.01 for i in range(10)],
                'Longitude': [-74.0060 - i*0.01 for i in range(10)],
                'Time Window Start': [
                    '08:00', '08:30', '09:00', '09:30', '10:00',
                    '10:30', '11:00', '11:30', '12:00', '12:30'
                ],
                'Time Window End': [
                    '10:00', '10:30', '11:00', '11:30', '12:00',
                    '12:30', '13:00', '13:30', '14:00', '14:30'
                ],
                'Service Time (min)': [15, 20, 10, 15, 25, 15, 20, 10, 15, 25],
                'Priority': ['High', 'Medium', 'Low', 'High', 'Medium', 
                            'Low', 'High', 'Medium', 'Low', 'High'],
                'Package Size': ['Small', 'Medium', 'Large', 'Small', 'Medium', 
                                'Large', 'Small', 'Medium', 'Large', 'Small']
            }
            st.session_state.df = pd.DataFrame(sample_data)
            st.success("Sample data loaded!")
    
    with col2:
        if st.button("Download Template"):
            template_df = generate_csv_template()
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv,
                file_name="delivery_template.csv",
                mime="text/csv"
            )

# Main area for data upload and processing
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("1️⃣ Upload Delivery Data")
    
    # File upload with clear instructions
    with st.expander("📋 CSV Format Instructions", expanded=False):
        st.markdown("""
        Your CSV file should include these columns:
        - **ID**: Unique identifier for each delivery
        - **Name**: Customer/delivery name
        - **Address**: Full address (used for geocoding if lat/lon not provided)
        - **Latitude**: Decimal latitude (optional if Address is provided)
        - **Longitude**: Decimal longitude (optional if Address is provided)
        - **Time Window Start**: Delivery start time in HH:MM format (optional)
        - **Time Window End**: Delivery end time in HH:MM format (optional)
        - **Service Time (min)**: Time in minutes needed at the location (optional)
        - **Priority**: High/Medium/Low (optional)
        - **Package Size**: Small/Medium/Large (optional)
        """)
    
    uploaded_file = st.file_uploader("📤 Upload your delivery data (CSV)", type=["csv"])

    # Load the data from the uploaded file
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate the uploaded data
            required_columns = ['ID', 'Name']
            location_columns = ['Latitude', 'Longitude']
            address_column = 'Address'
            
            # Check for required columns
            missing_required = [col for col in required_columns if col not in df.columns]
            
            # Check for location information (either coordinates or address)
            has_coordinates = all(col in df.columns for col in location_columns)
            has_address = address_column in df.columns
            
            if missing_required:
                st.error(f"Missing required columns: {', '.join(missing_required)}")
            elif not (has_coordinates or has_address):
                st.error("The CSV must contain either both Latitude & Longitude columns or an Address column.")
            else:
                # Validate the data
                validation_issues = validate_delivery_data(df)
                
                if validation_issues:
                    st.warning("Data validation issues found:")
                    for issue in validation_issues:
                        st.write(f"⚠️ {issue}")
                
                st.session_state.df = df
                st.success(f"✅ Successfully loaded {len(df)} delivery points")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    
    # Display the loaded data
    if st.session_state.df is not None:
        with st.expander("📦 View Delivery Data", expanded=False):
            st.dataframe(st.session_state.df, use_container_width=True)

with col2:
    st.subheader("2️⃣ Depot Location")
    depot_option = st.radio("Select depot location method:", 
                        ["Use first row in data as depot", "Enter depot coordinates manually"])
    
    if depot_option == "Enter depot coordinates manually":
        col_lat, col_lon = st.columns(2)
        with col_lat:
            depot_lat = st.number_input("Depot Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
        with col_lon:
            depot_lon = st.number_input("Depot Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
    else:
        if st.session_state.df is not None and 'Latitude' in st.session_state.df.columns and 'Longitude' in st.session_state.df.columns:
            depot_lat = st.session_state.df['Latitude'].iloc[0]
            depot_lon = st.session_state.df['Longitude'].iloc[0]
            st.info(f"Using first row as depot: ({depot_lat}, {depot_lon})")
        else:
            depot_lat = 40.7128
            depot_lon = -74.0060
            st.warning("No data loaded or missing coordinates. Using default depot location in NYC.")

# Route constraints and settings
st.subheader("3️⃣ Route Constraints & Settings")
col1, col2 = st.columns(2)

with col1:
    num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=10, value=3)
    
    # Vehicle capacity settings
    st.write("Vehicle Capacities")
    capacity_option = st.radio("Capacity Type", ["Uniform", "Custom"], horizontal=True)
    
    if capacity_option == "Uniform":
        uniform_capacity = st.number_input("Capacity (all vehicles)", min_value=1, max_value=100, value=5)
        vehicle_capacities = [uniform_capacity] * num_vehicles
    else:
        vehicle_capacities = []
        cols = st.columns(min(num_vehicles, 4))
        for i in range(num_vehicles):
            with cols[i % 4]:
                capacity = st.number_input(f"Vehicle {i+1}", min_value=1, max_value=100, value=5, key=f"vehicle_capacity_{i}")
                vehicle_capacities.append(capacity)

with col2:
    priority_weights = st.slider("Priority Importance (0-10)", min_value=0, max_value=10, value=5,
                              help="Higher values give more importance to delivery priorities")
    
    time_window_strictness = st.select_slider(
        "Time Window Strictness",
        options=["Flexible", "Moderate", "Strict"],
        value="Moderate",
        help="Determines how strictly to enforce time windows"
    )
    
    # Translate the time window strictness to a numeric value for the optimizer
    time_window_penalty = {
        "Flexible": 10,
        "Moderate": 50,
        "Strict": 200
    }[time_window_strictness]
    
    optimize_for = st.selectbox(
        "Optimize For",
        ["Minimize Total Distance", "Minimize Max Route Length", "Balance Routes"],
        index=0,
        help="Select the main optimization objective"
    )

# Additional constraints text area - FIXED EMPTY LABEL
st.write("Additional Constraints (Free Text for AI Context)")
additional_constraints = st.text_area(
    "Enter additional constraints",
    value="Avoid left turns when possible\nMaximize driver breaks during waiting times\nConsider traffic patterns",
    help="These constraints will be used for AI context but not directly in the optimization algorithm",
    label_visibility="collapsed"
)

# Run the optimization
if st.button("🧠 Optimize Routes", use_container_width=True):
    # Check if data is loaded
    if st.session_state.df is None:
        st.error("Please upload or load sample data first")
    else:
        with st.spinner("Processing data and calculating distances..."):
            # Handle the case where lat/long might be missing
            processor = DataProcessor()
            
            # If lat/long are missing but address is available, geocode the addresses
            if ('Latitude' not in st.session_state.df.columns or 'Longitude' not in st.session_state.df.columns) and 'Address' in st.session_state.df.columns:
                st.info("Geocoding addresses to get coordinates...")
                df_with_coords = processor.geocode_addresses(st.session_state.df)
                if df_with_coords is not None:
                    st.session_state.df = df_with_coords
                else:
                    st.error("Failed to geocode addresses. Please provide a CSV with Latitude and Longitude columns.")
                    st.stop()
            
            # Ensure required columns exist with default values if missing
            st.session_state.df = processor.ensure_required_columns(st.session_state.df)
            
            # Calculate distance matrix
            try:
                # Add depot to the beginning of locations if needed
                if depot_option == "Enter depot coordinates manually":                       
                    locations = [(depot_lat, depot_lon)] + list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                else:
                    locations = list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                
                # Validate locations
                if not locations or any(None in loc for loc in locations):
                    st.error("Invalid location data. Please check your coordinates.")
                    st.stop()
                
                # ALWAYS use Euclidean distances now
                st.info("Using straight-line distances...")
                distance_matrix = processor.calculate_euclidean_distance_matrix(locations)
                duration_matrix = distance_matrix * 0.12  # Simple time estimation (30 km/h)
                st.session_state.distance_matrix = distance_matrix
                st.session_state.duration_matrix = duration_matrix
                
            except Exception as e:
                st.error(f"Error calculating distances: {e}")
                st.stop()

            # Initialize the route optimizer
            optimizer = RouteOptimizer(
                distance_matrix=st.session_state.distance_matrix,
                duration_matrix=st.session_state.duration_matrix,
                num_vehicles=num_vehicles,
                depot=0,  # Assuming depot is always the first location
                vehicle_capacities=vehicle_capacities  # Pass capacities directly to constructor
            )
            
            # Prepare time windows if available
            time_windows = []
            if all(col in st.session_state.df.columns for col in ['Time Window Start', 'Time Window End']):
                # Convert time strings to minutes since midnight
                for _, row in st.session_state.df.iterrows():
                    try:
                        start_parts = row['Time Window Start'].split(':')
                        end_parts = row['Time Window End'].split(':')
                        
                        start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
                        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
                        
                        time_windows.append((start_minutes, end_minutes))
                    except:
                        # Use wide time window if parsing fails
                        time_windows.append((0, 24 * 60))
            else:
                # Default time windows (all day: 0 to 24 hours in minutes)
                time_windows = [(0, 24 * 60) for _ in range(len(st.session_state.df))]
            
            # Add depot time window (assume 24h operation for depot)
            time_windows.insert(0, (0, 24 * 60))
            
            # Prepare service times
            service_times = []
            if 'Service Time (min)' in st.session_state.df.columns:
                service_times = st.session_state.df['Service Time (min)'].fillna(0).astype(int).tolist()
            else:
                service_times = [15 for _ in range(len(st.session_state.df))]  # Default 15 minutes
            
            # Add zero service time for depot
            service_times.insert(0, 0)
            
            # Prepare priorities
            priorities = []
            if 'Priority' in st.session_state.df.columns:
                priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
                priorities = [priority_map.get(p, 1) for p in st.session_state.df['Priority'].fillna('Low')]
            else:
                priorities = [1 for _ in range(len(st.session_state.df))]  # Default priority
            
            # Add priority for depot (not used, but needs to match dimensions)
            priorities.insert(0, 0)
            
            # Set the parameters for the optimizer
            optimizer.set_time_windows(time_windows)
            optimizer.set_service_times(service_times)
            optimizer.set_priorities(priorities, priority_weights)
            
            # Choose the optimization strategy based on user selection
            if optimize_for == "Minimize Total Distance":
                optimizer.set_objective_minimize_total_distance()
            elif optimize_for == "Minimize Max Route Length":
                optimizer.set_objective_minimize_max_route()
            else:  # "Balance Routes"
                optimizer.set_objective_balance_routes()
            
            # Solve the optimization problem
            try:
                solution = optimizer.solve()
                if solution:
                    st.session_state.optimized_routes = solution
                else:
                    st.error("Could not find a solution. Try relaxing some constraints.")
                    st.stop()
            except Exception as e:
                st.error(f"Optimization error: {e}")
                st.stop()
        
        # Process the solution for display
        with st.spinner("Generating visualizations..."):
            if st.session_state.optimized_routes:
                # Generate the map visualization
                visualizer = MapVisualizer()
                
                # Extract locations for the map
                if depot_option == "Enter depot coordinates manually":
                    all_locations = [(depot_lat, depot_lon)] + list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                else:
                    all_locations = list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                
                # Generate map with the routes
                try:
                    folium_map = visualizer.create_route_map(
                        st.session_state.optimized_routes,
                        all_locations,
                        df=st.session_state.df
                    )
                    # Save the map HTML for later use
                    st.session_state.map_html = folium_map._repr_html_()
                except Exception as e:
                    st.error(f"Error creating map: {e}")
                    st.session_state.map_html = None
                
                # SKIP AI explanation - just set a placeholder
                st.session_state.ai_explanation = """
                ## Route Optimization Summary
                
                The routes have been optimized based on your selected criteria. 
                
                ### Key Benefits:
                - Routes are organized to minimize travel distance
                - Deliveries are grouped efficiently by geographic proximity
                - Time windows and priorities have been considered
                - Vehicle capacities are respected
                
                ### Notes:
                - This optimization uses straight-line distances
                - AI-powered explanations are temporarily disabled
                - Consider real-world factors like traffic and road conditions
                """
                    
                # Save the parameters used for this run
                st.session_state.last_run_params = {
                    "num_vehicles": num_vehicles,
                    "vehicle_capacities": vehicle_capacities,
                    "priority_weights": priority_weights,
                    "time_window_strictness": time_window_strictness,
                    "optimize_for": optimize_for,
                    "additional_constraints": additional_constraints
                }

# Display the results if available
if st.session_state.optimized_routes:
    st.subheader("4️⃣ Optimization Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Routes Overview", "Interactive Map", "Summary", "Export Options"])
    
    with tab1:
        # Display a summary of each route
        for vehicle_id, route in st.session_state.optimized_routes.items():
            # Skip empty routes
            if len(route) <= 2:  # Just depot-depot
                continue
                
            with st.expander(f"🚚 Vehicle {vehicle_id + 1} Route ({len(route) - 2} stops)", expanded=True):
                route_df = pd.DataFrame()
                
                # First add the depot
                if depot_option == "Enter depot coordinates manually":
                    depot_row = pd.DataFrame({
                        'Stop': [0],
                        'Type': ['Depot'],
                        'Name': ['Depot'],
                        'Address': ['Depot Location'],
                        'Arrival': ['Start'],
                        'Departure': ['Start']
                    })
                    route_df = pd.concat([route_df, depot_row], ignore_index=True)
                else:
                    depot_idx = route[0]
                    if depot_idx == 0:
                        # First row in original data is depot
                        depot_data = st.session_state.df.iloc[0:1].copy()
                        depot_data['Stop'] = 0
                        depot_data['Type'] = 'Depot'
                        depot_data['Arrival'] = 'Start'
                        depot_data['Departure'] = 'Start'
                        route_df = pd.concat([route_df, depot_data], ignore_index=True)
                
                # Add each stop in the route
                cumulative_time = 0  # in minutes
                for i, location_idx in enumerate(route[1:-1], 1):  # Skip depot at start and end
                    stop_idx = location_idx - 1  # Adjust for depot offset
                    stop_data = st.session_state.df.iloc[stop_idx:stop_idx+1].copy()
                    
                    # Add the stop number and type
                    stop_data['Stop'] = i
                    stop_data['Type'] = 'Delivery'
                    
                    # Calculate arrival time based on previous stop
                    if i > 1:
                        # Get travel time from previous stop to this stop
                        prev_idx = route[i-1]
                        travel_time = st.session_state.duration_matrix[prev_idx][location_idx] / 60  # Convert seconds to minutes
                        cumulative_time += travel_time
                    
                    # Format as HH:MM
                    start_time = datetime(2023, 1, 1, 8, 0, 0)  # Assume 8:00 AM start
                    arrival_time = start_time + timedelta(minutes=cumulative_time)
                    stop_data['Arrival'] = arrival_time.strftime('%H:%M')
                    
                    # Add service time
                    service_time = 15  # Default 15 minutes
                    if 'Service Time (min)' in stop_data.columns:
                        service_time = stop_data['Service Time (min)'].iloc[0]
                    
                    cumulative_time += service_time
                    departure_time = start_time + timedelta(minutes=cumulative_time)
                    stop_data['Departure'] = departure_time.strftime('%H:%M')
                    
                    route_df = pd.concat([route_df, stop_data], ignore_index=True)
                
                # Add depot return
                if len(route) > 2:  # Only add depot return if there are actual stops
                    if depot_option == "Enter depot coordinates manually":
                        # Get travel time from last stop to depot
                        last_idx = route[-2]  # Index of the last stop before returning to depot
                        travel_time = st.session_state.duration_matrix[last_idx][0] / 60  # Convert seconds to minutes
                        cumulative_time += travel_time
                        
                        return_time = start_time + timedelta(minutes=cumulative_time)
                        
                        depot_return_row = pd.DataFrame({
                            'Stop': [len(route)-1],
                            'Type': ['Depot Return'],
                            'Name': ['Depot'],
                            'Address': ['Depot Location'],
                            'Arrival': [return_time.strftime('%H:%M')],
                            'Departure': ['-']
                        })
                        route_df = pd.concat([route_df, depot_return_row], ignore_index=True)
                    else:
                        depot_idx = route[-1]
                        # Get travel time from last stop to depot
                        last_idx = route[-2]  # Index of the last stop before returning to depot
                        travel_time = st.session_state.duration_matrix[last_idx][depot_idx] / 60  # Convert seconds to minutes
                        cumulative_time += travel_time
                        
                        return_time = start_time + timedelta(minutes=cumulative_time)
                        
                        depot_data = st.session_state.df.iloc[0:1].copy()
                        depot_data['Stop'] = len(route)-1
                        depot_data['Type'] = 'Depot Return'
                        depot_data['Arrival'] = return_time.strftime('%H:%M')
                        depot_data['Departure'] = '-'
                        route_df = pd.concat([route_df, depot_data], ignore_index=True)
                
                # Calculate the total distance for this route
                route_distance = sum(st.session_state.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                
                # Create a nicer table for display
                display_cols = ['Stop', 'Type', 'Name', 'Address', 'Arrival', 'Departure']
                if 'Priority' in route_df.columns:
                    display_cols.append('Priority')
                if 'Package Size' in route_df.columns:
                    display_cols.append('Package Size')
                
                st.dataframe(route_df[display_cols], use_container_width=True)
                
                st.info(f"📏 Total Distance: {route_distance/1000:.2f} km  |  ⏱️ Estimated Duration: {cumulative_time:.0f} minutes")
    
    with tab2:
        # Display the interactive map
        if st.session_state.map_html:
            import streamlit.components.v1 as components
            components.html(st.session_state.map_html, height=600)
        else:
            st.info("Map visualization will appear here after optimization.")
    
    with tab3:
        # Display the summary
        if st.session_state.ai_explanation:
            st.markdown("### 📊 Route Optimization Summary")
            st.markdown(st.session_state.ai_explanation)
        else:
            st.info("Summary will appear here after optimization.")
    
    with tab4:
        # Export options
        st.markdown("### 📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export routes as CSV
            if st.button("📊 Export Routes as CSV", use_container_width=True):
                # Create a CSV with all route information
                all_routes_data = []
                
                for vehicle_id, route in st.session_state.optimized_routes.items():
                    if len(route) <= 2:  # Skip empty routes
                        continue
                    
                    for i, location_idx in enumerate(route):
                        if location_idx == 0:  # Depot
                            if i == 0:
                                stop_type = "Start (Depot)"
                            else:
                                stop_type = "End (Depot)"
                            row_data = {
                                'Vehicle': vehicle_id + 1,
                                'Stop_Number': i,
                                'Stop_Type': stop_type,
                                'Location': 'Depot',
                                'Name': 'Depot',
                                'Address': 'Depot Location'
                            }
                        else:
                            stop_idx = location_idx - 1
                            row = st.session_state.df.iloc[stop_idx]
                            row_data = {
                                'Vehicle': vehicle_id + 1,
                                'Stop_Number': i,
                                'Stop_Type': 'Delivery',
                                'Location': location_idx,
                                'Name': row['Name'],
                                'Address': row.get('Address', 'N/A'),
                                'Latitude': row['Latitude'],
                                'Longitude': row['Longitude'],
                                'Priority': row.get('Priority', 'N/A'),
                                'Package_Size': row.get('Package Size', 'N/A'),
                                'Time_Window_Start': row.get('Time Window Start', 'N/A'),
                                'Time_Window_End': row.get('Time Window End', 'N/A'),
                                'Service_Time': row.get('Service Time (min)', 'N/A')
                            }
                        
                        all_routes_data.append(row_data)
                
                # Convert to DataFrame and download
                routes_df = pd.DataFrame(all_routes_data)
                csv = routes_df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download Routes CSV",
                    data=csv,
                    file_name=f"optimized_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Generate PDF report
            if st.button("📄 Generate PDF Report", use_container_width=True):
                try:
                    from fpdf import FPDF
                    
                    # Create PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=16)
                    pdf.cell(0, 10, "Route Optimization Report", ln=True, align='C')
                    pdf.ln(5)
                    
                    # Add generation date
                    pdf.set_font("Arial", size=10)
                    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                    pdf.ln(5)
                    
                    # Add summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Summary", ln=True)
                    pdf.set_font("Arial", size=10)
                    
                    total_stops = sum(len(route) - 2 for route in st.session_state.optimized_routes.values() if len(route) > 2)
                    active_vehicles = sum(1 for route in st.session_state.optimized_routes.values() if len(route) > 2)
                    
                    pdf.cell(0, 8, f"Total Deliveries: {total_stops}", ln=True)
                    pdf.cell(0, 8, f"Active Vehicles: {active_vehicles} out of {num_vehicles}", ln=True)
                    pdf.cell(0, 8, f"Optimization Objective: {optimize_for}", ln=True)
                    pdf.ln(5)
                    
                    # Add routes
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Route Details", ln=True)
                    pdf.set_font("Arial", size=10)
                    
                    for vehicle_id, route in st.session_state.optimized_routes.items():
                        if len(route) <= 2:
                            continue
                        
                        pdf.ln(3)
                        pdf.set_font("Arial", 'B', 11)
                        pdf.cell(0, 8, f"Vehicle {vehicle_id + 1}", ln=True)
                        pdf.set_font("Arial", size=9)
                        
                        for i, location_idx in enumerate(route[1:-1], 1):
                            stop_idx = location_idx - 1
                            row = st.session_state.df.iloc[stop_idx]
                            pdf.cell(0, 6, f"  Stop {i}: {row['Name']} - {row.get('Address', 'N/A')}", ln=True)
                    
                    # Output PDF
                    pdf_output = pdf.output(dest='S').encode('latin1')
                    
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_output,
                        file_name=f"route_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
                    st.info("Make sure fpdf is installed: pip install fpdf")

# Add a footer with optimization statistics
if st.session_state.optimized_routes:
    st.divider()
    
    # Calculate statistics
    total_distance = 0
    total_stops = 0
    for vehicle_id, route in st.session_state.optimized_routes.items():
        if len(route) > 2:
            route_distance = sum(st.session_state.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            total_distance += route_distance
            total_stops += len(route) - 2
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Distance", f"{total_distance/1000:.1f} km")
    with col2:
        st.metric("Total Stops", total_stops)
    with col3:
        active_vehicles = sum(1 for route in st.session_state.optimized_routes.values() if len(route) > 2)
        st.metric("Active Vehicles", f"{active_vehicles}/{num_vehicles}")
    with col4:
        avg_stops = total_stops / active_vehicles if active_vehicles > 0 else 0
        st.metric("Avg Stops/Vehicle", f"{avg_stops:.1f}")

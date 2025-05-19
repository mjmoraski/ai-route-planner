import streamlit as st
import pandas as pd
import numpy as np
import openai
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

# Set up the page header and information
st.title("🚚 Final Mile AI Planner")
st.caption("🧠 Combines OR-Tools optimization + GenAI explanations with OpenStreetMap")

# Create sidebar for settings and configurations
with st.sidebar:
    st.header("Settings")
    
    api_option = st.radio("OpenAI API Key", ["Use from secrets.toml", "Enter API Key"], 
                         help="Choose where to get the OpenAI API key")
    
    if api_option == "Enter API Key":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
    else:
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            openai_api_key = ""
            st.warning("No API key found in secrets.toml")
    
    openai_model = st.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    st.subheader("Sample Data")
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
                capacity = st.number_input(f"Vehicle {i+1}", min_value=1, max_value=100, value=5)
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

# Additional constraints text area
st.write("Additional Constraints (Free Text for AI Context)")
additional_constraints = st.text_area(
    "",
    value="Avoid left turns when possible\nMaximize driver breaks during waiting times\nConsider traffic patterns",
    help="These constraints will be used for AI context but not directly in the optimization algorithm"
)

# Run the optimization
if st.button("🧠 Optimize Routes & Generate AI Explanation", use_container_width=True):
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
            
            # Calculate distance matrix using OSRM
            try:
                # Add depot to the beginning of locations
                if depot_option == "Enter depot coordinates manually":
                    locations = [(depot_lat, depot_lon)] + list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                else:
                    locations = list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                
                distance_matrix, duration_matrix = processor.calculate_distance_matrix(locations)
                st.session_state.distance_matrix = distance_matrix
                st.session_state.duration_matrix = duration_matrix
            except Exception as e:
                st.error(f"Error calculating distances: {e}")
                st.info("Falling back to straight-line distances...")
                
                # Fall back to Euclidean distances
                if depot_option == "Enter depot coordinates manually":
                    all_locations = [(depot_lat, depot_lon)] + list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                else:
                    all_locations = list(zip(st.session_state.df['Latitude'], st.session_state.df['Longitude']))
                
                distance_matrix = processor.calculate_euclidean_distance_matrix(all_locations)
                duration_matrix = [[dist * 2 for dist in row] for row in distance_matrix]  # Rough estimate: 30 km/h speed
                st.session_state.distance_matrix = distance_matrix
                st.session_state.duration_matrix = duration_matrix
        
        with st.spinner("Optimizing routes..."):
            # Set up the optimizer with all constraints and settings
            optimizer = RouteOptimizer(
                distance_matrix=st.session_state.distance_matrix,
                duration_matrix=st.session_state.duration_matrix,
                num_vehicles=num_vehicles,
                depot=0,  # Depot is the first location
                vehicle_capacities=vehicle_capacities
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
        with st.spinner("Generating visualizations and AI explanation..."):
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
                
                # Generate AI explanation
                try:
                    # Create a summary of the optimization results
                    routes_summary = []
                    total_distance = 0
                    max_route_length = 0
                    for vehicle_id, route in st.session_state.optimized_routes.items():
                        # Skip empty routes
                        if len(route) <= 2:  # Just depot-depot
                            continue
                            
                        route_distance = sum(st.session_state.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                        total_distance += route_distance
                        max_route_length = max(max_route_length, route_distance)
                        
                        # Create a list of stops on this route
                        stops = []
                        for location_idx in route[1:-1]:  # Skip depot at start and end
                            stop_idx = location_idx - 1  # Adjust for depot offset
                            row = st.session_state.df.iloc[stop_idx]
                            
                            stop_info = {
                                'ID': row['ID'],
                                'Name': row['Name'],
                                'Address': row.get('Address', 'N/A'),
                                'Priority': row.get('Priority', 'N/A'),
                                'Time Window': f"{row.get('Time Window Start', 'N/A')} - {row.get('Time Window End', 'N/A')}",
                                'Service Time': f"{row.get('Service Time (min)', 'N/A')} minutes"
                            }
                            stops.append(stop_info)
                        
                        routes_summary.append({
                            'Vehicle': vehicle_id + 1,
                            'Stops': len(stops),
                            'Distance': f"{route_distance/1000:.2f} km",
                            'Stops_List': stops
                        })
                    
                    # Create the prompt for the AI explanation
                    prompt = f"""You are a logistics expert. Explain the following optimized delivery routes in business-friendly terms.
Focus on efficiency, customer service benefits, and practical advantages for drivers.

Optimization Parameters:
- Number of Vehicles: {num_vehicles}
- Vehicle Capacities: {vehicle_capacities}
- Priority Importance (0-10): {priority_weights}
- Time Window Strictness: {time_window_strictness}
- Optimization Objective: {optimize_for}
- Additional Constraints: {additional_constraints}

Routes Summary:
Total Distance: {total_distance/1000:.2f} km
Max Route Length: {max_route_length/1000:.2f} km
Number of Routes: {len([r for r in routes_summary if len(r['Stops_List']) > 0])}

Detailed Routes:
{json.dumps(routes_summary, indent=2)}

Your explanation should:
1. Highlight the key benefits of this route plan
2. Explain how the routes respect the constraints
3. Provide insights on the balance between vehicles
4. Mention any special considerations for drivers
5. Suggest potential improvements for future routes

Keep your explanation clear, practical, and focused on business value.
"""

                    # Call OpenAI
                    client = openai.OpenAI(api_key=openai_api_key)
                    
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )
                    
                    st.session_state.ai_explanation = response.choices[0].message.content
                    
                except Exception as e:
                    st.error(f"Error generating AI explanation: {e}")
                    st.session_state.ai_explanation = "Could not generate AI explanation. Please check your OpenAI API key and try again."
                    
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
    tab1, tab2, tab3, tab4 = st.tabs(["Routes Overview", "Interactive Map", "AI Explanation", "Export Options"])
    
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
            # Get routes for detailed display
            routes_for_map = []
            for vehicle_id, route in st.session_state.optimized_routes.items():
                if len(route) > 2:  # Skip empty routes
                    route_locations = []
                    for idx in route:
                        if idx == 0:  # Depot
                            if depot_option == "Enter depot coordinates manually":
                                route_locations.append((depot_lat, depot_lon))
                            else:
                                route_locations.append((st.session_state.df['Latitude'].iloc[0], st.session_state.df['Longitude'].iloc[0]))
                        else:
                            route_locations.append((st.session_state.df['Latitude'].iloc[idx-1], st.session_state.df['Longitude'].iloc[idx-1]))
                    
                    routes_for_map.append({
                        'vehicle_id': vehicle_id,
                        'locations': route_locations
                    })
            
            # Create a fresh map
            visualizer = MapVisualizer()
            map_center = [sum(loc[0] for loc in all_locations) / len(all_locations),
                          sum(loc[1] for loc in all_locations) / len(all_locations)]
            
            route_map = visualizer.create_route_map(
                st.session_state.optimized_routes,
                all_locations,
                df=st.session_state.df
            )
            
            folium_static(route_map, width=800, height=600)
        else:
            st.warning("Could not generate map visualization. Try again or check your data.")
    
    with tab3:
        # Display the AI explanation
        if st.session_state.ai_explanation:
            st.markdown(st.session_state.ai_explanation)
        else:
            st.warning("No AI explanation available. Please run the optimization with a valid OpenAI API key.")
    
    with tab4:
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to CSV
            if st.button("📄 Export Routes to CSV", use_container_width=True):
                # Create a DataFrame with all routes
                export_dfs = []
                
                for vehicle_id, route in st.session_state.optimized_routes.items():
                    if len(route) <= 2:  # Skip empty routes
                        continue
                        
                    # Create a route dataframe
                    route_export = []
                    
                    for i, location_idx in enumerate(route):
                        if location_idx == 0:  # Depot
                            if i == 0:  # Starting at depot
                                route_export.append({
                                    'Vehicle': f"Vehicle {vehicle_id + 1}",
                                    'Stop': 0,
                                    'Type': 'Depot Start',
                                    'ID': 'DEPOT',
                                    'Name': 'Depot',
                                    'Address': 'Depot Location',
                                    'Latitude': depot_lat if depot_option == "Enter depot coordinates manually" else st.session_state.df['Latitude'].iloc[0],
                                    'Longitude': depot_lon if depot_option == "Enter depot coordinates manually" else st.session_state.df['Longitude'].iloc[0]
                                })
                            else:  # Returning to depot
                                route_export.append({
                                    'Vehicle': f"Vehicle {vehicle_id + 1}",
                                    'Stop': i,
                                    'Type': 'Depot Return',
                                    'ID': 'DEPOT',
                                    'Name': 'Depot',
                                    'Address': 'Depot Location',
                                    'Latitude': depot_lat if depot_option == "Enter depot coordinates manually" else st.session_state.df['Latitude'].iloc[0],
                                    'Longitude': depot_lon if depot_option == "Enter depot coordinates manually" else st.session_state.df['Longitude'].iloc[0]
                                })
                        else:
                            # Regular stop
                            stop_idx = location_idx - 1  # Adjust for depot offset
                            row = st.session_state.df.iloc[stop_idx]
                            
                            stop_info = {
                                'Vehicle': f"Vehicle {vehicle_id + 1}",
                                'Stop': i,
                                'Type': 'Delivery',
                                'ID': row['ID'],
                                'Name': row['Name'],
                                'Address': row.get('Address', 'N/A'),
                                'Latitude': row['Latitude'],
                                'Longitude': row['Longitude']
                            }
                            
                            # Add optional fields if available
                            for field in ['Priority', 'Time Window Start', 'Time Window End', 'Service Time (min)', 'Package Size']:
                                if field in row:
                                    stop_info[field] = row[field]
                            
                            route_export.append(stop_info)
                    
                    export_dfs.append(pd.DataFrame(route_export))
                
                # Combine all routes
                if export_dfs:
                    all_routes_df = pd.concat(export_dfs, ignore_index=True)
                    
                    # Convert to CSV
                    csv = all_routes_df.to_csv(index=False)
                    
                    # Create download link
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="optimized_routes.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("CSV ready for download!")
                else:
                    st.error("No routes to export!")
        
        with col2:
            # Generate PDF report
            if st.button("📊 Generate Driver Reports (PDF)", use_container_width=True):
                with st.spinner("Generating PDF reports..."):
                    # Create a BytesIO object for the PDF
                    pdf_buffer = io.BytesIO()
                    
                    # Create a PDF document
                    class PDF(FPDF):
                        def header(self):
                            self.set_font('Arial', 'B', 15)
                            self.cell(0, 10, 'Route Plan - Driver Report', 0, 1, 'C')
                            self.ln(5)
                        
                        def footer(self):
                            self.set_y(-15)
                            self.set_font('Arial', 'I', 8)
                            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                    
                    # Create a PDF for each route
                    for vehicle_id, route in st.session_state.optimized_routes.items():
                        if len(route) <= 2:  # Skip empty routes
                            continue
                            
                        pdf = PDF()
                        pdf.add_page()
                        
                        # Add title and date
                        pdf.set_font('Arial', 'B', 16)
                        pdf.cell(0, 10, f"Route Plan - Vehicle {vehicle_id + 1}", ln=True)
                        pdf.set_font('Arial', '', 12)
                        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
                        
                        # Add route summary
                        route_distance = sum(st.session_state.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, "Route Summary:", ln=True)
                        pdf.set_font('Arial', '', 12)
                        pdf.cell(0, 10, f"Number of Stops: {len(route) - 2}", ln=True)
                        pdf.cell(0, 10, f"Total Distance: {route_distance/1000:.2f} km", ln=True)
                        pdf.cell(0, 10, f"Estimated Duration: {sum(st.session_state.duration_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) / 60:.0f} minutes", ln=True)
                        
                        # Add stops table
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, "Delivery Sequence:", ln=True)
                        
                        # Table header
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(15, 10, "Stop", 1)
                        pdf.cell(60, 10, "Customer", 1)
                        pdf.cell(75, 10, "Address", 1)
                        pdf.cell(40, 10, "Time Window", 1)
                        pdf.ln()
                        
                        # Table content
                        pdf.set_font('Arial', '', 10)
                        
                        # Start at depot
                        pdf.cell(15, 10, "0", 1)
                        pdf.cell(60, 10, "DEPOT", 1)
                        pdf.cell(75, 10, "Depot Location", 1)
                        pdf.cell(40, 10, "Start", 1)
                        pdf.ln()
                        
                        # Add each stop
                        for i, location_idx in enumerate(route[1:-1], 1):
                            stop_idx = location_idx - 1  # Adjust for depot offset
                            row = st.session_state.df.iloc[stop_idx]
                            
                            customer_name = row['Name']
                            address = row.get('Address', 'N/A')
                            
                            time_window = "Any time"
                            if all(col in row for col in ['Time Window Start', 'Time Window End']):
                                time_window = f"{row['Time Window Start']} - {row['Time Window End']}"
                            
                            pdf.cell(15, 10, str(i), 1)
                            pdf.cell(60, 10, customer_name, 1)
                            pdf.cell(75, 10, address[:35] + "..." if len(address) > 35 else address, 1)
                            pdf.cell(40, 10, time_window, 1)
                            pdf.ln()
                        
                        # End at depot
                        pdf.cell(15, 10, str(len(route)-1), 1)
                        pdf.cell(60, 10, "DEPOT", 1)
                        pdf.cell(75, 10, "Depot Location", 1)
                        pdf.cell(40, 10, "End", 1)
                        pdf.ln(20)
                        
                        # Add notes section
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, "Notes:", ln=True)
                        pdf.set_font('Arial', '', 10)
                        pdf.multi_cell(0, 10, "Use this space for any additional information or special instructions for the driver.")
                        
                        # Add the PDF to the buffer
                        pdf_buffer.write(pdf.output(dest='S').encode('latin1'))
                    
                    # Create download link
                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="driver_reports.pdf">Download PDF Reports</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("PDF reports ready for download!")
          

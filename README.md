# AI Route Planner

## Overview
An advanced route optimization application that combines Google OR-Tools for route optimization with OpenAI's GPT models for intelligent route explanations. The application visualizes optimized delivery routes on interactive maps using OpenStreetMap APIs.

## Features
- **Real Distance Matrix**: Uses OpenStreetMap's OSRM API for realistic travel times and distances
- **Multiple Vehicles**: Support for multiple vehicles with customizable capacities
- **Advanced Constraints**: 
  - Time windows for deliveries
  - Service times at each location
  - Priority-based sequencing
  - Package size considerations
- **Interactive Map Visualization**: See optimized routes on a dynamic map
- **AI-Powered Explanations**: Get business-friendly explanations of why routes are optimal
- **Export Options**: 
  - Export routes to CSV
  - Generate PDF driver reports
- **Geocoding**: Automatically convert addresses to coordinates when needed

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Streamlit
- OR-Tools
- OpenAI API key
- Internet connection for API access

### Installation
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Create a `.streamlit/secrets.toml` file with your OpenAI API key: OPENAI_API_KEY = "your-api-key-here"

### Running the Application
Start the application with:
streamlit run app.py

## Usage
1. **Upload Data**: Upload your CSV file with delivery information
2. **Set Depot**: Specify where vehicles start and end their routes
3. **Configure Routes**: Set the number of vehicles and their capacities
4. **Adjust Constraints**: Customize time windows, priorities, and more
5. **Optimize**: Calculate the optimal routes and get AI explanations
6. **Export**: Download route information as CSV or PDF

## CSV Format
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

## Technical Details

### Optimization Algorithm
The app uses Google OR-Tools' Vehicle Routing Problem (VRP) solver with:
- Custom distance and time matrices
- Time window constraints
- Priority-based sequencing
- Multiple vehicle support
- Capacity constraints

### Map Integration
- Uses OpenStreetMap through the OSRM API
- Visualizes routes using Folium

### AI Integration
- Uses OpenAI's API to generate natural language explanations
- Highlights efficiency, customer service benefits, and driver advantages


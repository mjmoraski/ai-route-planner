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
2. Install dependencies:

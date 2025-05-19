import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Truck, Clock, ArrowRight, Package, Flag, Calendar } from 'lucide-react';

const RouteDashboard = () => {
  const [routeSummary, setRouteSummary] = useState([]);
  const [priorityData, setPriorityData] = useState([]);
  const [totalDistance, setTotalDistance] = useState(0);
  const [totalStops, setTotalStops] = useState(0);
  const [loadingData, setLoadingData] = useState(true);
  const [selectedVehicle, setSelectedVehicle] = useState(null);

  // Colors for the charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];
  const PRIORITY_COLORS = {
    'High': '#ff4d4f',
    'Medium': '#faad14',
    'Low': '#52c41a'
  };

  // Simulated data loading - in a real app, this would come from the backend
  useEffect(() => {
    // Simulate data loading delay
    setTimeout(() => {
      const sampleRouteData = [
        { vehicle: 'Vehicle 1', stops: 5, distance: 18.5, maxLoad: 80 },
        { vehicle: 'Vehicle 2', stops: 4, distance: 15.2, maxLoad: 65 },
        { vehicle: 'Vehicle 3', stops: 3, distance: 10.8, maxLoad: 45 }
      ];
      
      const samplePriorityData = [
        { name: 'High', value: 4 },
        { name: 'Medium', value: 5 },
        { name: 'Low', value: 3 }
      ];
      
      setRouteSummary(sampleRouteData);
      setPriorityData(samplePriorityData);
      setTotalDistance(sampleRouteData.reduce((acc, route) => acc + route.distance, 0));
      setTotalStops(sampleRouteData.reduce((acc, route) => acc + route.stops, 0));
      setLoadingData(false);
    }, 1000);
  }, []);

  // Detailed info for selected vehicle
  const getVehicleDetails = () => {
    if (!selectedVehicle) return null;
    
    // In a real implementation, this would fetch specific data for the selected vehicle
    const vehicleIndex = parseInt(selectedVehicle.split(' ')[1]) - 1;
    const vehicleData = routeSummary[vehicleIndex];
    
    return (
      <div className="bg-white rounded-lg shadow p-4 mt-4">
        <h3 className="text-lg font-semibold mb-2">{selectedVehicle} Details</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-gray-600">Estimated Start Time</p>
            <p className="text-lg font-medium">8:00 AM</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-gray-600">Estimated End Time</p>
            <p className="text-lg font-medium">2:30 PM</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-gray-600">Total Drive Time</p>
            <p className="text-lg font-medium">4h 15m</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-gray-600">Service Time</p>
            <p className="text-lg font-medium">2h 15m</p>
          </div>
        </div>
        
        <h4 className="text-md font-semibold mt-4 mb-2">Stop Sequence</h4>
        <div className="space-y-2">
          {[...Array(vehicleData.stops + 1)].map((_, i) => (
            <div key={i} className="flex items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${i === 0 || i === vehicleData.stops ? 'bg-gray-800 text-white' : 'bg-blue-500 text-white'}`}>
                {i + 1}
              </div>
              <ArrowRight className="mx-2" size={16} />
              <div className="bg-gray-50 flex-1 p-2 rounded">
                {i === 0 || i === vehicleData.stops ? 'Depot' : `Customer ${(vehicleIndex * 5) + i}`}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loadingData) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-lg text-gray-600">Loading route data...</div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Route Optimization Dashboard</h2>
      
      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-blue-100 p-3 mr-4">
            <Truck className="text-blue-600" size={24} />
          </div>
          <div>
            <p className="text-gray-500">Total Vehicles</p>
            <p className="text-2xl font-bold">{routeSummary.length}</p>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-green-100 p-3 mr-4">
            <Package className="text-green-600" size={24} />
          </div>
          <div>
            <p className="text-gray-500">Total Stops</p>
            <p className="text-2xl font-bold">{totalStops}</p>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-amber-100 p-3 mr-4">
            <Flag className="text-amber-600" size={24} />
          </div>
          <div>
            <p className="text-gray-500">Total Distance</p>
            <p className="text-2xl font-bold">{totalDistance.toFixed(1)} km</p>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-purple-100 p-3 mr-4">
            <Clock className="text-purple-600" size={24} />
          </div>
          <div>
            <p className="text-gray-500">Est. Completion</p>
            <p className="text-2xl font-bold">3:45 PM</p>
          </div>
        </div>
      </div>
      
      {/* Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-2">Distance by Vehicle</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={routeSummary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="vehicle" />
                <YAxis label={{ value: 'Distance (km)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="distance" fill="#0088FE" name="Distance (km)" onClick={(data) => setSelectedVehicle(data.vehicle)} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-2">Stops by Priority</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={priorityData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {priorityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={PRIORITY_COLORS[entry.name] || COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Vehicle details if selected */}
      {selectedVehicle && getVehicleDetails()}
      
      {/* Route Table */}
      <div className="bg-white rounded-lg shadow p-4 mt-6">
        <h3 className="text-lg font-semibold mb-2">Route Summary</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-2 px-4 border-b text-left">Vehicle</th>
                <th className="py-2 px-4 border-b text-left">Stops</th>
                <th className="py-2 px-4 border-b text-left">Total Distance</th>
                <th className="py-2 px-4 border-b text-left">Max Load</th>
                <th className="py-2 px-4 border-b text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {routeSummary.map((route, index) => (
                <tr key={index} className={selectedVehicle === route.vehicle ? "bg-blue-50" : ""}>
                  <td className="py-2 px-4 border-b">{route.vehicle}</td>
                  <td className="py-2 px-4 border-b">{route.stops}</td>
                  <td className="py-2 px-4 border-b">{route.distance} km</td>
                  <td className="py-2 px-4 border-b">{route.maxLoad}%</td>
                  <td className="py-2 px-4 border-b">
                    <button 
                      className="bg-blue-500 hover:bg-blue-700 text-white py-1 px-3 rounded"
                      onClick={() => setSelectedVehicle(route.vehicle)}
                    >
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default RouteDashboard;

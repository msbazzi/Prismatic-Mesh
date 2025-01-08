import pyvista as pv
from pyvista import examples
import numpy as np
from scipy.interpolate import interp1d

# Load the lumen mesh
heat_transfer_results = pv.read('heat_transfer/result_002.vtu')
inner_wall = pv.read('lumen-mesh-complete/mesh-surfaces/lumen_wall.vtp')

points_wall = inner_wall.points
points_heat_transfer = heat_transfer_results.points
temperature_values = heat_transfer_results.point_data['Temperature']  # Replace 'Temperature' with the actual name if different

# Step 2: Define the known temperature-thickness relationship
temperatures_known = np.array([0, 3.33, 6.66, 10])  # Example known temperature values
thickness_values_known = np.array([0.753, 0.756, 0.816, 0.825])  # Corresponding thickness values

# Create the interpolation function for thickness based on temperature
thickness_interp = interp1d(temperatures_known, thickness_values_known, kind='linear', fill_value="extrapolate")

threshold_distance = 0.01 

inner_wall_indices = []
for i, point in enumerate(points_heat_transfer):
    # Find the closest point in the inner wall
    dist = np.linalg.norm(points_wall - point, axis=1)
    if np.min(dist) < threshold_distance:
        inner_wall_indices.append(i)



# Step 3: Extract the temperature values of the points on the inner wall
temperature_values_inner_wall = heat_transfer_results.point_data['Temperature'][inner_wall_indices]

# Step 4: Interpolate the thickness values for the selected nodes on the inner wall
node_thicknesses = thickness_interp(temperature_values_inner_wall)

# Step 5: Optionally, visualize the distribution of thicknesses for only the inner wall points
inner_wall.point_data['Thickness'] = node_thicknesses  # Add thickness as a new point data field

# Step 5: Optionally, visualize the distribution of thicknesses across the mesh

# Plot the mesh with the thickness as the scalar value
plotter = pv.Plotter()
plotter.add_mesh(inner_wall, scalars='Thickness', cmap='viridis', show_edges=True)
plotter.show()
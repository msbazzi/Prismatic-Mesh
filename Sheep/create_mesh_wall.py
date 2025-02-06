import vtk
import pyvista as pv
import numpy as np
import os
from utils import *
from scipy.spatial import KDTree
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def extract_faces(unstructured_grid, output_file):
    # Use vtkGeometryFilter to extract the surface
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()

    # Get the output surface as vtkPolyData
    surface = geometry_filter.GetOutput()

        # Check if GlobalNodeID exists
    global_node_id_array = unstructured_grid.GetPointData().GetArray("GlobalNodeID")
    if global_node_id_array:
        print("GlobalNodeID found. Propagating to the surface...")
        surface.GetPointData().AddArray(global_node_id_array)
    else:
        print("GlobalNodeID not found in the volumetric mesh.")

    # Write the surface to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(surface)
    writer.Write()

    #print(f"Extracted surface written to: {output_file}")
    return surface

def point_normals(surface):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(surface)
    reader.Update()
    polydata = reader.GetOutput()

    # Validate input data
    num_points = polydata.GetNumberOfPoints()
    num_cells = polydata.GetNumberOfCells()

    # print(f"Number of points: {num_points}")
    # print(f"Number of cells: {num_cells}")

    if num_points == 0 or num_cells == 0:
        print("The mesh is empty or invalid. Check the .vtp file.")
        exit()  # Stop execution if the file is invalid

    # Check for pre-computed normals
    if polydata.GetPointData().GetNormals():
        print("Normals already exist in the file.")
        normals = polydata.GetPointData().GetNormals()
    else:
        print("Normals not found in the file. Computing normals...")
        # Compute normals
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()  # Focus only on point normals
        normals_filter.Update()

        # Get the output with computed normals
        polydata = normals_filter.GetOutput()

    # Validate normals after computation
    normals = polydata.GetPointData().GetNormals()
    if normals is None:
        print("Failed to compute normals.")
        exit()  # Stop if normals computation failed
    else:
        print("Normals computed successfully. Number of points with normals:",  polydata.GetNumberOfPoints())

    # Convert to PyVista
    mesh = pv.wrap(polydata)

    if mesh is None or mesh.n_points == 0:
        print("The mesh is empty after wrapping with PyVista.")
        exit()

    if "Normals" not in mesh.point_data:
        print("Normals not found in the PyVista mesh.")
        exit()

    count = np.zeros(mesh.points.shape[0], dtype=int)
    # Assuming mesh is already loaded with incorrect normals
    if mesh['Normals'].shape[0] != mesh.points.shape[0]:
        del mesh.cell_data['Normals']

        # Assuming each point should only have one unique normal but appears to have multiple
        # We will create a corrected normals array based on unique points
        unique_points, indices = np.unique(mesh.points, return_inverse=True, axis=0)
        corrected_normals = np.zeros_like(unique_points)

        for i, idx in enumerate(indices):
            corrected_normals[idx] += mesh['Normals'][i]
            count[idx] += 1

        # Average the normals where duplicates were added
        corrected_normals /= np.maximum(1, count[:, None])  # Avoid division by zero

        mesh['Normals'] = corrected_normals
        print(f"Corrected normals shape: {mesh['Normals'].shape}")
        
        # if mesh.points.shape == mesh["Normals"].shape:
        #     plotter = pv.Plotter()
        #     plotter.add_arrows(mesh.points, mesh["Normals"], mag=0.1, color="red")
        #     plotter.show()
        # else:
        #     print("Error: The shapes of mesh.points and mesh['Normals'] still do not match.")
        #     print(f"mesh.points shape: {mesh.points.shape}")
        #     print(f"mesh['Normals'] shape: {mesh['Normals'].shape}")

    normal_point = np.array([normals.GetTuple(i) for i in range(mesh.n_points)])
    mesh["Normals"] = normal_point
    mesh["Z_Normals"] = normal_point[:, 2]


    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh,  scalars="Z_Normals", show_edges=True, opacity=0.99)  # Add surface
    # #plotter.add_arrows(mesh.points, mesh["Normals"], mag=0.1, color="red")  # Add normals
    # plotter.show()

    return polydata, normals  

def plot_points(points):
    # Create a PyVista point cloud from the NumPy array
    point_cloud = pv.PolyData(points)

    # Plot the points
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="blue", point_size=5, render_points_as_spheres=True)
    plotter.show()

def get_axial_positions(polydata):
    num_points = polydata.GetNumberOfPoints()
    axial_positions = np.zeros(num_points)
    
    for i in range(num_points):
        point = polydata.GetPoint(i)
        axial_positions[i] = point[2]  # Assuming the z-coordinate is the axial position

    return axial_positions

def generate_prismatic_mesh_vtu_nonuniform_thickness(polydata, normals, heat_transfer_results, thickness_vector, radial_layers, output_file):
    # Create an array to store the new points
    
    num_points = polydata.GetNumberOfPoints()
    points_wall = polydata.GetPoints().GetData()
    all_points = np.zeros(((radial_layers+1)*num_points, 7))

    points_heat_transfer = heat_transfer_results.points
    temperature_values = heat_transfer_results.point_data['Temperature'] 
    temperatures_known = np.linspace(np.min(temperature_values), np.max(temperature_values), len(thickness_vector))
    thickness_values_known = thickness_vector
    # Smooth input thickness values
    smoothed_thickness = gaussian_filter1d(thickness_values_known, sigma=1)

    thickness_interp = interp1d(temperatures_known, smoothed_thickness, kind='cubic', fill_value="extrapolate")

    threshold_distance = 0.01 

    tree = KDTree(points_wall)

    distances, indices = tree.query(points_heat_transfer)

    inner_wall_indices = np.where(distances < threshold_distance)[0]
 
    temperature_values_inner_wall = heat_transfer_results.point_data['Temperature'][inner_wall_indices]

    node_thicknesses = thickness_interp(temperature_values_inner_wall)

    z_positions = points_heat_transfer[:, 2]
    outlet_threshold = z_positions.max() - 0.05  # Define the threshold for the last 5% of the domain
    outlet_region = z_positions > outlet_threshold
    node_thicknesses[outlet_region] = np.linspace(
    node_thicknesses[outlet_region].min(),
    node_thicknesses[outlet_region].mean(),
    outlet_region.sum())

    inner_wall_points = points_heat_transfer[inner_wall_indices]  

    inner_wall_mesh = pv.PolyData(inner_wall_points)


    inner_wall_mesh.point_data['Thickness'] = node_thicknesses

    # Plot the mesh with the thickness as the scalar value
    plotter = pv.Plotter()
    plotter.add_mesh(inner_wall_mesh, scalars='Thickness', cmap='viridis', show_edges=True)
    plotter.show()
    
    ids = np.zeros(((radial_layers+1)*num_points),  dtype=int)
    prism_connectivity = []
    global_id_counter = 0
    inner_ids = np.zeros(((radial_layers+1)*num_points), dtype=int)
    outer_ids = np.zeros(((radial_layers+1)*num_points), dtype=int)

    for i in range(num_points):

        normal = normals.GetTuple(i)
        global_id_array = polydata.GetPointData().GetArray("GlobalNodeID").GetValue(i)
        points_coordinates = polydata.GetPoint(i)
        all_points[i,0]= global_id_array
        all_points[i,1] = points_coordinates[0]
        all_points[i,2] = points_coordinates[1]
        all_points[i,3] = points_coordinates[2]
        all_points[i,4] = normal[0]
        all_points[i,5] = normal[1]
        all_points[i,6] = normal[2]
        inner_ids[i] = 1
        ids[i] = 1


    num_cells = polydata.GetNumberOfCells()
    cell_point_ids = [global_id_counter + i for i in range(num_points)]
    
    # Generate new points for each radial layer
   
    for i in range(radial_layers):
        c = i+1
        factor = c / (radial_layers)  # Scale factor for radial layers

        for j in range(num_points):
            thickness = node_thicknesses[j] * factor
            point = polydata.GetPoint(j)
            normal = normals.GetTuple(j)

            # Calculate the offset point
            xPt = point[0] +  thickness * (normal[0])
            yPt = point[1] +  thickness * (normal[1])
            zPt = point[2] +  thickness * (normal[2])
            
            all_points[num_points*c+j,0] = all_points[num_points*i+j,0]+num_points*c
            all_points[num_points*c+j,1] = xPt
            all_points[num_points*c+j,2] = yPt
            all_points[num_points*c+j,3] = zPt
            all_points[num_points*c+j,4] = normal[0]
            all_points[num_points*c+j,5] = normal[1]
            all_points[num_points*c+j,6] = normal[2]
        
            if i == radial_layers - 1:
                outer_ids[num_points*c+j] = 1
                ids[num_points*c+j] = 2
    
    # Generate connectivity for prismatic cells
    for i in range(num_cells):
        
        cell = polydata.GetCell(i)
        num_cell_points = cell.GetNumberOfPoints()
        cell_point_ids = [cell.GetPointId(j) for j in range(num_cell_points)]

        for j in range(radial_layers):
            base_layer = j * num_points
            next_layer = (j + 1) * num_points

            # Create prismatic connectivity
            if num_cell_points==3: # Triangular cells
                p0 = cell_point_ids[0] + base_layer
                p1 = cell_point_ids[1] + base_layer
                p2 = cell_point_ids[2] + base_layer

                p3 = cell_point_ids[0] + next_layer
                p4 = cell_point_ids[1] + next_layer
                p5 = cell_point_ids[2] + next_layer
                # Add a prism cell: 6 points per cell
                prism_connectivity.append([p0, p1, p2, p3, p4, p5])
             
            elif len(num_cell_points)==4: # Quadrilateral cells
                print("Quadrilateral cells is not implemented yet")              
                    # Convert all_points to a NumPy array
       
    # Convert points to VTK format
    all_points_coordinates  = all_points[:,1:4]
    #plot_points(all_points_coordinates)
    vtk_points = vtk.vtkPoints()
    for point in all_points_coordinates:
        vtk_points.InsertNextPoint(point)

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(vtk_points)

    # Add prism cells to the unstructured grid
    for prism in prism_connectivity:
        vtk_cell = vtk.vtkIdList()
        for pid in prism:
            vtk_cell.InsertNextId(pid)
        unstructured_grid.InsertNextCell(vtk.VTK_WEDGE, vtk_cell)
    # Write to VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    # vol_py= pv.wrap(unstructured_grid)
    # plotter = pv.Plotter()
    # plotter.add_mesh(vol_py, scalars="OuterID", show_edges=True)
    # plotter.show()
    
    print("VTU file written to: ", output_file)

    return unstructured_grid, all_points

def generate_prismatic_mesh_vtu(polydata, normals, thickness, radial_layers, output_file):
    # Create an array to store the new points
    
    num_points = polydata.GetNumberOfPoints()
    all_points = np.zeros(((radial_layers+1)*num_points, 7))
    ids = np.zeros(((radial_layers+1)*num_points),  dtype=int)
    prism_connectivity = []
    global_id_counter = 0
    inner_ids = np.zeros(((radial_layers+1)*num_points), dtype=int)
    outer_ids = np.zeros(((radial_layers+1)*num_points), dtype=int)

    for i in range(num_points):

        normal = normals.GetTuple(i)
        global_id_array = polydata.GetPointData().GetArray("GlobalNodeID").GetValue(i)
        points_coordinates = polydata.GetPoint(i)
        all_points[i,0]= global_id_array
        all_points[i,1] = points_coordinates[0]
        all_points[i,2] = points_coordinates[1]
        all_points[i,3] = points_coordinates[2]
        all_points[i,4] = normal[0]
        all_points[i,5] = normal[1]
        all_points[i,6] = normal[2]
        inner_ids[i] = 1
        ids[i] = 1


    num_cells = polydata.GetNumberOfCells()
    cell_point_ids = [global_id_counter + i for i in range(num_points)]
    
    # Generate new points for each radial layer
   
    for i in range(radial_layers):
        c = i+1
        factor = c / (radial_layers)  # Scale factor for radial layers
        
        for j in range(num_points):

            point = polydata.GetPoint(j)
            normal = normals.GetTuple(j)

            # Calculate the offset point
            xPt = point[0] + factor * thickness * (normal[0])
            yPt = point[1] + factor * thickness * (normal[1])
            zPt = point[2] + factor * thickness * (normal[2])
            
            all_points[num_points*c+j,0] = all_points[num_points*i+j,0]+num_points*c
            all_points[num_points*c+j,1] = xPt
            all_points[num_points*c+j,2] = yPt
            all_points[num_points*c+j,3] = zPt
            all_points[num_points*c+j,4] = normal[0]
            all_points[num_points*c+j,5] = normal[1]
            all_points[num_points*c+j,6] = normal[2]
        
            if i == radial_layers - 1:
                outer_ids[num_points*c+j] = 1
                ids[num_points*c+j] = 2
    
    # Generate connectivity for prismatic cells
    for i in range(num_cells):
        
        cell = polydata.GetCell(i)
        num_cell_points = cell.GetNumberOfPoints()
        cell_point_ids = [cell.GetPointId(j) for j in range(num_cell_points)]

        for j in range(radial_layers):
            base_layer = j * num_points
            next_layer = (j + 1) * num_points

            # Create prismatic connectivity
            if num_cell_points==3: # Triangular cells
                p0 = cell_point_ids[0] + base_layer
                p1 = cell_point_ids[1] + base_layer
                p2 = cell_point_ids[2] + base_layer

                p3 = cell_point_ids[0] + next_layer
                p4 = cell_point_ids[1] + next_layer
                p5 = cell_point_ids[2] + next_layer
                # Add a prism cell: 6 points per cell
                prism_connectivity.append([p0, p1, p2, p3, p4, p5])
             
            elif len(num_cell_points)==4: # Quadrilateral cells
                print("Quadrilateral cells is not implemented yet")              
                    # Convert all_points to a NumPy array
    
    
   
    # Convert points to VTK format
    all_points_coordinates  = all_points[:,1:4]
    #plot_points(all_points_coordinates)
    vtk_points = vtk.vtkPoints()
    for point in all_points_coordinates:
        vtk_points.InsertNextPoint(point)

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(vtk_points)

    # Add prism cells to the unstructured grid
    for prism in prism_connectivity:
        vtk_cell = vtk.vtkIdList()
        for pid in prism:
            vtk_cell.InsertNextId(pid)
        unstructured_grid.InsertNextCell(vtk.VTK_WEDGE, vtk_cell)
    # Write to VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    # vol_py= pv.wrap(unstructured_grid)
    # plotter = pv.Plotter()
    # plotter.add_mesh(vol_py, scalars="OuterID", show_edges=True)
    # plotter.show()
    
    print("VTU file written to: ", output_file)

    return unstructured_grid, all_points

def thresholdModelNew(data, array_name, min_value, max_value):
    thresh_filter = vtk.vtkThreshold()
    thresh_filter.SetInputData(data)
    # Set to process cell data
    thresh_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, array_name)
    thresh_filter.SetLowerThreshold(min_value)
    thresh_filter.SetUpperThreshold(max_value)
    thresh_filter.Update()
    return thresh_filter.GetOutput()

def convert_to_polydata(unstructured_grid):
    """Convert vtkUnstructuredGrid to vtkPolyData."""
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()
    return geometry_filter.GetOutput()

def saveSolid(vol,surf,output_dir):
      
        os.makedirs(output_dir + '/mesh-surfaces', exist_ok=True)
        save_data(output_dir + '/mesh-complete.mesh.vtu',vol)
        save_data(output_dir + '/mesh-complete.mesh.vtp',surf)

        proximal = thresholdModelNew(surf, "ModelFaceID",0.5,1.5)
        proximal_polydata = convert_to_polydata(proximal)  # Convert to vtkPolyData
        save_data(output_dir + '/mesh-surfaces/wall_inner.vtp',proximal_polydata)

        inner = thresholdModelNew(surf, 'ModelFaceID',1.5,2.5)
        inner_polydata = convert_to_polydata(inner)
        save_data(output_dir + '/mesh-surfaces/wall_inlet.vtp',inner_polydata)

        distal = thresholdModelNew(surf, "ModelFaceID", 2.5, 3.5)
        distal_polydata = convert_to_polydata(distal)
        save_data(output_dir + '/mesh-surfaces/wall_outlet.vtp', distal_polydata)

        outer = thresholdModelNew(surf, "ModelFaceID", 3.5, 4.5)
        outer_polydata = convert_to_polydata(outer)
        save_data(output_dir + '/mesh-surfaces/wall_outer.vtp', outer_polydata)
        if outer_polydata.GetNumberOfPoints() == 0:
            print("No outer surface found.")
            return
        else:
            print("Outer surface saved to: ", output_dir + '/mesh-surfaces/wall_outer.vtp')

        return

def run_script_with_sv(python_sv,extract_faces_script, faces_file, output_dir):
    command = [
        python_sv, 
        "--python", 
        "--", 
        extract_faces_script, 
        faces_file, 
        output_dir
    ]

    try:
        # Run the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Print the output of the command
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Error Output:\n", e.stderr)

def read_pathlines(pathlines_file):
# Load and parse the XML file
    tree = ET.parse(pathlines_file)
    root = tree.getroot()

    # Extract control points
    control_points = []
    for point in root.findall(".//control_points/point"):
        x = float(point.get("x"))
        y = float(point.get("y"))
        z = float(point.get("z"))
        control_points.append((x, y, z))

    # Extract path points
    path_points = []
    for path_point in root.findall(".//path_points/path_point"):
        pos = path_point.find("pos")
        x = float(pos.get("x"))
        y = float(pos.get("y"))
        z = float(pos.get("z"))
        path_points.append((x, y, z))

    # Example output
    print("Control Points (first 5):", control_points[:5])
    print("Path Points (first 5):", path_points[:5])

    return control_points, path_points
    
def main():

    current_directory = os.getcwd()
    python_sv = "/home/bazzi/repo/SimVascular-Ubuntu-20-2023-05/data/usr/local/sv/simvascular/2023-03-27/simvascular"
    extract_faces_script = current_directory + "/get_faces_with_sv.py"

    # heat_transfer_results = pv.read('heat_transfer/result_002.vtu')
    # pathlines_file = current_directory + "/Sheep/IVC.pth"
    surface = current_directory + "/mesh/refined_es_006/fluid/mesh-surfaces/lumen_wall.vtp"
    output_file = current_directory + "/mesh/refined_es_006/solid/wall.vtu"
    faces_file = current_directory +  "/mesh/refined_es_006/solid/wall_faces.vtp"
    output_dir = current_directory + "/mesh/refined_es_006/solid"
    
    radial_layers = 4
    thickness_vector = np.mean([0.119, 0.113, 0.109, 0.109])

    # #extrat the normals to each point in the surface and read the face mesh 
    polydata, normals = point_normals(surface)
    
    # # use the normals to generate a prismatic mesh starting from the surface mesh (polydata) return the volume mesh (vol)
    
    vol, all_points = generate_prismatic_mesh_vtu(polydata, normals, thickness_vector, radial_layers, output_file) #thickness_vector is a scalar value
    #vol, all_points =  generate_prismatic_mesh_vtu_nonuniform_thickness(polydata, normals, heat_transfer_results, thickness_vector, radial_layers, output_file)

    # # Add GlobalNodeID and GlobalElementID to the volume mesh
    global_ids = all_points[:,0]
    coordinates = all_points[:,1:4]
    numCells = vol.GetNumberOfCells()
    vol.GetPointData().AddArray(pv.convert_array(coordinates.astype(float),name="Coordinate"))
    vol.GetPointData().AddArray(pv.convert_array(global_ids.astype(np.int32),name="GlobalNodeID"))
    vol.GetCellData().AddArray(pv.convert_array(np.linspace(1,numCells,numCells).astype(np.int32),name="GlobalElementID"))

    # #extract the faces without ids from the volume mesh (vol) and save it to wall_faces.vtp
    extract_faces(vol, faces_file)

    # #run the python wrap in SimVascular to extract the faces from the volume mesh (vol) and save it to wall_faces_with_ids.vtp"
    run_script_with_sv(python_sv, extract_faces_script, faces_file, output_dir)

    # #read faces mesh (surfaces_with_ids) 
    surfaces_with_ids = pv.read(output_dir + "/wall_faces_with_ids.vtp")
    
    # # split and save all the faces in the surface-mesh and the volume mesh as mesh-complete.mesh.vtu
    saveSolid(vol,surfaces_with_ids,output_dir)

if __name__ == "__main__":
    main()
import vtk
import pyvista as pv
import numpy as np
import os
from utils import *
from scipy.spatial import KDTree
import subprocess


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
        save_data(output_dir + '/mesh-surfaces/wall_inlet.vtp',proximal_polydata)
        if proximal_polydata.GetNumberOfPoints() == 0:
            print("No inlet surface found.")
            return
        else:
            print("Inlet surface saved to: ", output_dir + '/mesh-surfaces/wall_inlet.vtp')

        inner = thresholdModelNew(surf, 'ModelFaceID',1.5,2.5)
        inner_polydata = convert_to_polydata(inner)
        save_data(output_dir + '/mesh-surfaces/wall_inner.vtp',inner_polydata)
        if inner_polydata.GetNumberOfPoints() == 0:
            print("No inner surface found.")
            return
        else:
            print("Inner surface saved to: ", output_dir + '/mesh-surfaces/wall_inner.vtp')

        distal = thresholdModelNew(surf, "ModelFaceID", 2.5, 3.5)
        distal_polydata = convert_to_polydata(distal)
        save_data(output_dir + '/mesh-surfaces/wall_outlet.vtp', distal_polydata)
        if distal_polydata.GetNumberOfPoints() == 0:
            print("No outlet surface found.")
            return
        else:
            print("Outlet surface saved to: ", output_dir + '/mesh-surfaces/wall_outlet.vtp')   

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

def main():
    python_sv = "/home/bazzi/repo/SimVascular-Ubuntu-20-2023-05/data/usr/local/sv/simvascular/2023-03-27/simvascular"
    extract_faces_script = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/get_faces_with_sv.py"
    
    surface = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/lumen-mesh-complete/mesh-surfaces/lumen_wall.vtp"
    output_file="/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete/wall.vtu"
    faces_file = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete/wall_faces.vtp"
    output_dir = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete"
    
    #extrat the normals to each point in the surface and read the face mesh 
    polydata, normals = point_normals(surface)
    
    # use the normals to generate a prismatic mesh starting from the surface mesh (polydata) return the volume mesh (vol)
    vol, all_points = generate_prismatic_mesh_vtu(polydata, normals, 0.1, 4, output_file)

    # Add GlobalNodeID and GlobalElementID to the volume mesh
    global_ids = all_points[:,0]
    coordinates = all_points[:,1:4]
    numCells = vol.GetNumberOfCells()
    vol.GetPointData().AddArray(pv.convert_array(coordinates.astype(float),name="Coordinate"))
    vol.GetPointData().AddArray(pv.convert_array(global_ids.astype(np.int32),name="GlobalNodeID"))
    vol.GetCellData().AddArray(pv.convert_array(np.linspace(1,numCells,numCells).astype(np.int32),name="GlobalElementID"))

    #extract the faces without ids from the volume mesh (vol) and save it to wall_faces.vtp
    extract_faces(vol, faces_file)

    #run the python wrap in SimVascular to extract the faces from the volume mesh (vol) and save it to wall_faces_with_ids.vtp"
    run_script_with_sv(python_sv, extract_faces_script, faces_file, output_dir)

    #read faces mesh (surfaces_with_ids) 
    surfaces_with_ids = pv.read(output_dir + "/wall_faces_with_ids.vtp")
    
    # split and save all the faces in the surface-mesh and the volume mesh as mesh-complete.mesh.vtu
    saveSolid(vol,surfaces_with_ids,output_dir)

if __name__ == "__main__":
    main()
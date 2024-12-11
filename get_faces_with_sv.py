import sv
import vtk
import numpy as np
import os
from scipy.spatial import KDTree


def get_faces(file,output_dir):
    
    kernel = sv.modeling.Kernel.POLYDATA
    modeler = sv.modeling.Modeler(kernel)
    model = modeler.read(file)
    print("Model type: " + str(type(model)))

    try:
        face_ids = model.get_face_ids()
    except:
        face_ids = model.compute_boundary_faces(angle=60.0)
    print("Model face IDs: " + str(face_ids))


    file_name = "/home/bazzi/TEVG/Prismatic_mesh/solid-mesh-complete/faces"
    file_format = "vtp"
    model.write(file_name=file_name, format=file_format)

    return face_ids

def main():
    surface = "/home/bazzi/TEVG/Prismatic_mesh/lumen-mesh-complete/mesh-surfaces/lumen_wall.vtp"
    output_file="/home/bazzi/TEVG/Prismatic_mesh/solid-mesh-complete/wall.vtu"
    faces_file = "/home/bazzi/TEVG/Prismatic_mesh/solid-mesh-complete/wall_faces.vtp"
    output_dir = "/home/bazzi/TEVG/Prismatic_mesh/solid-mesh-complete"

    faces_id = get_faces(faces_file,output_dir)
    

if __name__ == "__main__":
    main()
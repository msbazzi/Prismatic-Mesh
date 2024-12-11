import sv
import argparse

def get_faces(file,output_dir):
    
    kernel = sv.modeling.Kernel.POLYDATA
    modeler = sv.modeling.Modeler(kernel)
    model = modeler.read(file)
    print("Model type: " + str(type(model)))

    face_ids = model.compute_boundary_faces(angle=60.0)
    print("Model face IDs: " + str(face_ids))


    file_name = output_dir + "/wall_faces_with_ids" #"/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete/faces"
    print("File name: " + file_name)
    file_format = "vtp"
    model.write(file_name=file_name, format=file_format)

    return face_ids

def main():
    # faces_file = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete/wall_faces.vtp"
    # output_dir = "/home/bazzi/TEVG/Prismatic_mesh/Sheep/solid-mesh-complete"

    parser = argparse.ArgumentParser(description="Compute and save model faces.")
    parser.add_argument("faces_file", type=str, help="Path to the input VTP file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()
    print("Faces file: " + args.faces_file)

    get_faces(args.faces_file, args.output_dir)
    
if __name__ == "__main__":
    main()

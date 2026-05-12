import pyvista as pv
import os
import glob

def batch_convert_directory(input_dir, output_dir):
    # 1. Ensure the output directory exists
    # exist_ok=True means it won't crash if the folder is already there
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    # 2. Look for all .obj files in the input folder
    search_pattern = os.path.join(input_dir, "*.OBJ")
    obj_files = glob.glob(search_pattern)
    
    if len(obj_files) == 0:
        print(f"No .obj files found in {input_dir}")
        return
        
    print(f"Found {len(obj_files)} .obj files. Starting batch conversion...\n")
    
    success_count = 0
    fail_count = 0

    # 3. Loop through every file
    for obj_path in obj_files:
        # Extract just the filename without the extension (e.g., "molar_36")
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        
        # Build the exact save path in the new output folder
        stl_path = os.path.join(output_dir, base_name + ".stl")
        
        file_name = os.path.basename(obj_path)
        print(f"Converting: {file_name} -> {base_name}.stl")
        
        try:
            # Read the OBJ and save it as a Binary STL in the new folder
            mesh = pv.read(obj_path)
            mesh.save(stl_path, binary=True)
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to convert {file_name}: {e}")
            fail_count += 1

    # 4. Print the final report
    print("\n--- BATCH PROCESS COMPLETE ---")
    print(f"Successfully converted: {success_count}")
    if fail_count > 0:
        print(f"Failed conversions: {fail_count}")
    print(f"All STL files have been securely saved to: {output_dir}")

# Define your library paths
input_folder = "/home/shiva/Documents/NuDent/Tooth_Library/tooth_obj"
output_folder = "/home/shiva/Documents/NuDent/Tooth_Library/tooth_stl"

# Run the batch processor
batch_convert_directory(input_folder, output_folder)
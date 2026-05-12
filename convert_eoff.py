import numpy as np
import trimesh
import os

def crack_eoff_smart(input_path, output_path):
    print(f"Analyzing byte structure of {input_path}...")
    
    with open(input_path, 'rb') as f:
        # 1. Read header and counts
        header = f.readline()
        counts = np.fromfile(f, dtype=np.uint32, count=3)
        v_count, f_count, e_count = counts[0], counts[1], counts[2]
        
        # 2. Read the entire rest of the file to see exactly how big it is
        remaining_bytes = f.read()
        actual_size = len(remaining_bytes)
        
        print(f"Geometry expected -> Vertices: {v_count}, Faces: {f_count}")
        print(f"Payload byte size -> {actual_size} bytes")
        
        # --- THE REVERSE-ENGINEERING LOGIC ---
        # Scenario 1: Standard Binary (12 bytes per vertex + 16 bytes per face)
        size_standard = (v_count * 12) + (f_count * 16)
        
        # Scenario 2: Stripped Binary (12 bytes per vertex + 12 bytes per face)
        size_stripped = (v_count * 12) + (f_count * 12)

        # Scenario 3: High Precision Float64 (24 bytes per vertex + 16 bytes per face)
        size_float64 = (v_count * 24) + (f_count * 16)

        if actual_size == size_standard:
            print("Format detected: Standard Binary OFF")
            v_dtype, f_cols, has_count = np.float32, 4, True
        elif actual_size == size_stripped:
            print("Format detected: Stripped Binary (No Face Counts)")
            v_dtype, f_cols, has_count = np.float32, 3, False
        elif actual_size == size_float64:
            print("Format detected: High-Precision (Float64 Vertices)")
            v_dtype, f_cols, has_count = np.float64, 4, True
        else:
            print(f"Sizes -> Standard: {size_standard}, Stripped: {size_stripped}, Float64: {size_float64}")
            raise ValueError(f"Unknown format! Actual size ({actual_size}) doesn't match any known dental format.")

    # 3. Now that we know their secret layout, read it perfectly
    with open(input_path, 'rb') as f:
        f.readline() # skip header
        np.fromfile(f, dtype=np.uint32, count=3) # skip counts
        
        # Read vertices
        vertices = np.fromfile(f, dtype=v_dtype, count=v_count * 3).reshape((-1, 3))
        
        # Read faces
        faces_raw = np.fromfile(f, dtype=np.uint32, count=f_count * f_cols).reshape((-1, f_cols))
        
        if has_count:
            faces = faces_raw[:, 1:4] # Drop the '3' column
        else:
            faces = faces_raw # Use directly
            
    # 4. Compile and save the STL
    print("Compiling 3D mesh...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)
    print(f"Success! Clean STL saved to {output_path}")
    
    # Render the result
    mesh.show()

# Run it
file_in = "/home/shiva/Documents/NuDent/M.O.D Rectangular/teeth/M.O.D Rectangular/upperjaw/1.eoff"
file_out = "tooth_1_cracked.stl"

crack_eoff_smart(file_in, file_out)
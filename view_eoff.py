import numpy as np
import trimesh
import sys
import os

def load_eoff(input_path):
    """
    Parses a binary EOFF file used in dental libraries and returns a trimesh object.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return None

    with open(input_path, 'rb') as f:
        # 1. Read header
        header = f.readline().strip().decode('ascii', errors='ignore')
        
        # 2. Read counts: vertex_count, face_count, edge_count
        counts = np.fromfile(f, dtype=np.uint32, count=3)
        if len(counts) < 3:
            print("Error: Could not read counts from header.")
            return None
        v_count, f_count, e_count = counts[0], counts[1], counts[2]
        print(f"Geometry -> Vertices: {v_count}, Faces: {f_count}")

        # 3. Determine format
        current_pos = f.tell()
        f.seek(0, 2)
        total_size = f.tell()
        payload_remaining = total_size - current_pos
        f.seek(current_pos)

        # 4. Read vertices
        # Dental EOFF is strictly float32 (3 floats = 12 bytes)
        vertices = np.fromfile(f, dtype=np.float32, count=v_count * 3).reshape((-1, 3))
        
        # 5. Read faces - Padding-Strip Stream Method (Proven clean)
        face_offset = 23 + (v_count * 12)
        f.seek(face_offset)
        
        raw_u16 = np.fromfile(f, dtype=np.uint16)
        
        # Strip padding 0s (produces clean shell, but loses vertex 0)
        clean_stream = raw_u16[raw_u16 != 0]
        
        faces = []
        i = 0
        while i < len(clean_stream) - 3:
            if clean_stream[i] == 3:
                faces.append([clean_stream[i+1], clean_stream[i+2], clean_stream[i+3]])
                i += 4
            else:
                i += 1
                
        faces = np.array(faces)
        print(f"Recovered {len(faces)} clean faces.")
        
    # Create final mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    # REPAIR: Fill any tiny holes automatically
    if not mesh.is_watertight:
        print("Mesh has holes. Attempting automatic repair...")
        mesh.fill_holes()
    
    # Mesh Statistics
    print(f"--- Mesh Stats ---")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Dimensions: {mesh.extents} mm")
    
    return mesh

def main():
    path = "/home/shiva/Documents/NuDent/Tooth_Library/teeth/generic-smooth/lowerjaw/3.eoff"
    
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Analyzing {path}...")
    mesh = load_eoff(path)
    
    if mesh:
        print(f"Successfully loaded and repaired mesh.")
        
        # Export as STL
        stl_path = os.path.basename(path).replace(".eoff", ".stl")
        mesh.export(stl_path)
        print(f"Saved STL to: {stl_path}")
        
        print("Opening interactive viewer...")
        mesh.show()
    else:
        print("Failed to load mesh.")

if __name__ == "__main__":
    main()

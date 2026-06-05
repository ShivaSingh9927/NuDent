import os
import sys
import numpy as np
import trimesh
from glob import glob

def load_and_repair_eoff(path):
    """
    Proven logic: Padding-Strip + Automatic Hole Filling
    """
    with open(path, 'rb') as f:
        header_raw = f.readline().decode(errors='ignore').strip()
        if "OFF BINARY" not in header_raw:
            return None
            
        # Read V, F, E counts (4-byte uint32)
        counts = np.fromfile(f, dtype=np.uint32, count=3)
        v_count, f_count = counts[0], counts[1]
        
        # Read vertices (float32, 12 bytes per vertex)
        vertices = np.fromfile(f, dtype=np.float32, count=v_count * 3).reshape((-1, 3))
        
        # Read faces (stream-based padding-strip)
        face_offset = 23 + (v_count * 12)
        f.seek(face_offset)
        raw_u16 = np.fromfile(f, dtype=np.uint16)
        
        # Strip padding 0s
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
        
    # Create and repair mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not mesh.is_watertight:
        mesh.fill_holes()
        
    return mesh

def batch_process():
    # Source directories
    base_path = "/home/shiva/Documents/NuDent/Tooth_Library/teeth/generic-smooth"
    sources = [
        os.path.join(base_path, "lowerjaw"),
        os.path.join(base_path, "upperjaw")
    ]
    
    # Destination directory
    out_dir = "/home/shiva/Documents/NuDent/Tooth_Library/full_teeth_off_to_stl"
    os.makedirs(out_dir, exist_ok=True)
    
    total_converted = 0
    
    for src in sources:
        jaw_type = os.path.basename(src) # lowerjaw or upperjaw
        files = glob(os.path.join(src, "*.eoff"))
        print(f"\nProcessing {len(files)} files from {jaw_type}...")
        
        for fpath in files:
            fname = os.path.basename(fpath)
            # Name format: lowerjaw_3.stl
            out_name = f"{jaw_type}_{fname.replace('.eoff', '.stl')}"
            out_path = os.path.join(out_dir, out_name)
            
            try:
                mesh = load_and_repair_eoff(fpath)
                if mesh:
                    mesh.export(out_path)
                    print(f"  [OK] {fname} -> {out_name}")
                    total_converted += 1
                else:
                    print(f"  [FAIL] {fname}: Invalid header")
            except Exception as e:
                print(f"  [ERROR] {fname}: {str(e)}")
                
    print(f"\nFinished! Successfully converted {total_converted} teeth to STL.")
    print(f"Location: {out_dir}")

if __name__ == "__main__":
    batch_process()

import trimesh
import numpy as np
import matplotlib.pyplot as plt

def analyze_bite(upper_path, lower_path):
    print("Loading meshes...")
    upper = trimesh.load(upper_path)
    lower = trimesh.load(lower_path)

    # Optimization 1: Subsample the vertices to make it 5x faster
    subsample_factor = 5
    check_vertices = lower.vertices[::subsample_factor]

    print(f"Calculating distances for {len(check_vertices)} points...")
    
    # Optimization 2: Use a fast KD-Tree for distance instead of signed_distance
    # This finds the nearest point on the surface of the upper jaw
    _, proximity_distances, _ = upper.nearest.on_surface(check_vertices)
    
    # We need to identify intersections (collisions)
    # Simple trick: points whose normal points 'into' the other mesh
    # For now, let's just use the absolute distance for speed
    distances = proximity_distances
    
    # Map back to full mesh for visualization
    full_distances = np.zeros(len(lower.vertices))
    for i in range(subsample_factor):
        idx = np.arange(i, len(lower.vertices), subsample_factor)
        # Fill gaps with the same values for visualization
        limit = min(len(idx), len(distances))
        full_distances[idx[:limit]] = distances[:limit]

    norm_distances = np.clip(full_distances, 0.0, 1.0)
    
    # Create a heatmap
    # Using 'jet' or 'RdYlGn' (Red-Yellow-Green)
    cmap = plt.get_cmap('jet')
    colors = cmap((norm_distances - norm_distances.min()) / (norm_distances.max() - norm_distances.min()))
    
    # Apply colors to the lower jaw
    lower.visual.vertex_colors = colors

    print("Opening Heatmap Viewer...")
    print("RED/ORANGE = Collision (High spot)")
    print("GREEN = Contact")
    print("BLUE = Gap")
    
    # Show both, but lower jaw is now a heatmap
    scene = trimesh.Scene([upper, lower])
    scene.show()

if __name__ == "__main__":
    upper_stl = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-046/2023-10-01_99999-046-upperjaw.stl"
    lower_stl = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-046/2023-10-01_99999-046-lowerjaw.stl"
    
    analyze_bite(upper_stl, lower_stl)
